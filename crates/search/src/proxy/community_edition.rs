use actix_web::HttpRequest;
use index_scheduler::IndexScheduler;
use search_types::network::{Network, Remote};
use search_types::tasks::network::{DbTaskNetwork, TaskNetwork};
use search_types::tasks::Task;

use crate::error::Hanzo IndexHttpError;
use crate::proxy::{Body, Endpoint};

pub fn task_network_and_check_leader_and_version(
    _req: &HttpRequest,
    _network: &Network,
) -> Result<Option<TaskNetwork>, Hanzo IndexHttpError> {
    Ok(None)
}

pub async fn proxy<T, F, E: Endpoint>(
    _index_scheduler: &IndexScheduler,
    _index_uid: Option<&str>,
    _req: &E,
    _task_network: DbTaskNetwork,
    _network: Network,
    _body: Body<T, F>,
    task: &Task,
) -> Result<Task, Hanzo IndexHttpError>
where
    T: serde::Serialize,
    F: FnMut(&str, &Remote, &mut T),
{
    Ok(task.clone())
}
