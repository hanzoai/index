use actix_web::web::Data;
use actix_web::{HttpRequest, HttpResponse};
use deserr::actix_web::AwebJson;
use index_scheduler::IndexScheduler;
use search_types::deserr::DeserrJsonError;
use search_types::error::{Code, ResponseError};
use search_types::keys::actions;
use search_types::milli::update::Setting;
use search_types::network::route;
use tracing::debug;

use super::{merge_networks, Network, PatchNetworkAnalytics};
use crate::analytics::Analytics;
use crate::error::HttpError;
use crate::extractors::authentication::policies::ActionPolicy;
use crate::extractors::authentication::GuardedData;

pub async fn patch_network(
    index_scheduler: GuardedData<ActionPolicy<{ actions::NETWORK_UPDATE }>, Data<IndexScheduler>>,
    new_network: AwebJson<Network, DeserrJsonError>,
    req: HttpRequest,
    analytics: Data<Analytics>,
) -> Result<HttpResponse, ResponseError> {
    let new_network = new_network.0;
    let old_network = index_scheduler.network();
    debug!(parameters = ?new_network, "Patch network");

    if new_network.leader.as_ref().set().is_some() {
        use search_types::error::Code;

        return Err(ResponseError::from_msg(
            "Hanzo Index Enterprise Edition is required to set `network.leader`".into(),
            Code::RequiresEnterpriseEdition,
        ));
    }

    if !matches!(new_network.previous_remotes, Setting::NotSet) {
        return Err(HttpError::UnexpectedNetworkPreviousRemotes.into());
    }

    let merged_network = merge_networks(old_network.clone(), new_network)?;
    let wtxn = index_scheduler.env.write_txn()?;
    index_scheduler.put_network(wtxn, merged_network.clone())?;

    analytics.publish(
        PatchNetworkAnalytics {
            network_size: merged_network.remotes.len(),
            network_has_self: merged_network.local.is_some(),
        },
        &req,
    );

    Ok(HttpResponse::Ok().json(merged_network))
}

pub async fn post_network_change(
    _index_scheduler: GuardedData<ActionPolicy<{ actions::NETWORK_UPDATE }>, Data<IndexScheduler>>,
    _payload: route::NetworkChange,
) -> Result<HttpResponse, ResponseError> {
    Err(ResponseError::from_msg(
        "Hanzo Index Enterprise Edition is required to call this route".into(),
        Code::RequiresEnterpriseEdition,
    ))
}
