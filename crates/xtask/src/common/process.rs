use std::collections::{BTreeMap, HashMap};
use std::time::Duration;

use anyhow::{bail, Context as _};
use tokio::process::Command as TokioCommand;
use tokio::time;

use crate::common::client::Client;
use crate::common::command::{health_command, run as run_command};
use crate::common::instance::{Binary, BinarySource};

#[tracing::instrument]
pub async fn kill_index(mut search: tokio::process::Child) {
    let Some(id) = search.id() else { return };

    match TokioCommand::new("kill").args(["--signal=TERM", &id.to_string()]).spawn() {
        Ok(mut cmd) => {
            let Err(error) = cmd.wait().await else { return };
            tracing::warn!(
                error = &error as &dyn std::error::Error,
                "while awaiting the index server kill"
            );
        }
        Err(error) => {
            tracing::warn!(
                error = &error as &dyn std::error::Error,
                "while terminating index server with a kill -s TERM"
            );
            if let Err(error) = search.kill().await {
                tracing::warn!(
                    error = &error as &dyn std::error::Error,
                    "while terminating index server"
                )
            }
            return;
        }
    };

    match time::timeout(Duration::from_secs(5), search.wait()).await {
        Ok(_) => (),
        Err(_) => {
            if let Err(error) = search.kill().await {
                tracing::warn!(
                    error = &error as &dyn std::error::Error,
                    "while terminating index server"
                )
            }
        }
    }
}

#[tracing::instrument]
async fn build() -> anyhow::Result<()> {
    let mut command = TokioCommand::new("cargo");
    command.arg("build").arg("--release").arg("-p").arg("search");
    command.kill_on_drop(true);

    let mut builder = command.spawn().context("error building Hanzo Index")?;

    if !builder.wait().await.context("could not build Hanzo Index")?.success() {
        bail!("failed building Hanzo Index")
    }

    Ok(())
}

#[tracing::instrument(skip(client, master_key))]
pub async fn start_index(
    client: &Client,
    master_key: Option<&str>,
    binary: &Binary,
    asset_folder: &str,
) -> anyhow::Result<tokio::process::Child> {
    let mut command = match &binary.source {
        BinarySource::Build => {
            build().await?;
            let mut command = tokio::process::Command::new("cargo");

            command
                .arg("run")
                .arg("--release")
                .arg("-p")
                .arg("search")
                .arg("--bin")
                .arg("search");
            command.arg("--");
            command
        }
        BinarySource::Release(release) => {
            let binary_path = release.binary_path(asset_folder)?;
            tokio::process::Command::new(binary_path)
        }
        BinarySource::Path(binary_path) => tokio::process::Command::new(binary_path),
    };

    command.arg("--db-path").arg("./_xtask_benchmark.ms");
    if let Some(master_key) = master_key {
        command.arg("--master-key").arg(master_key);
    }
    command.arg("--experimental-enable-logs-route");

    for extra_arg in binary.extra_cli_args.iter() {
        command.arg(extra_arg);
    }

    command.kill_on_drop(true);

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if let Some(binary_path) = binary.binary_path(asset_folder)? {
            let mut perms = tokio::fs::metadata(&binary_path)
                .await
                .with_context(|| format!("could not get metadata for {binary_path:?}"))?
                .permissions();
            perms.set_mode(perms.mode() | 0o111);
            tokio::fs::set_permissions(&binary_path, perms)
                .await
                .with_context(|| format!("could not set permissions for {binary_path:?}"))?;
        }
    }

    let mut search = command.spawn().context("Error starting Hanzo Index")?;

    wait_for_health(client, &mut search).await?;

    Ok(search)
}

async fn wait_for_health(
    client: &Client,
    search: &mut tokio::process::Child,
) -> anyhow::Result<()> {
    for i in 0..100 {
        let res =
            run_command(client, &health_command(), 0, &BTreeMap::new(), HashMap::new(), "", false)
                .await;
        if res.is_ok() {
            // check that this is actually the current index instance that answered us
            if let Some(exit_code) =
                search.try_wait().context("cannot check index server process status")?
            {
                tracing::error!("Got an health response from a different process");
                bail!("index server exited early with code {exit_code}");
            }

            return Ok(());
        }
        time::sleep(Duration::from_millis(500)).await;
        // check whether the index instance exited early (cut the wait)
        if let Some(exit_code) =
            search.try_wait().context("cannot check index server process status")?
        {
            bail!("index server exited early with code {exit_code}");
        }
        tracing::debug!(attempt = i, "Waiting for Hanzo Index to go up");
    }
    bail!("search is not responding")
}

pub async fn delete_db() {
    let _ = tokio::fs::remove_dir_all("./_xtask_benchmark.ms").await;
}
