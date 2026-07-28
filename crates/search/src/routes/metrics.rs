use actix_web::web::{self, Data};
use actix_web::HttpResponse;
use index_scheduler::{IndexScheduler, Query};
use search_auth::AuthController;
use search_types::deserr::query_params::Param;
use search_types::error::ResponseError;
use search_types::keys::actions;
use search_types::milli::progress::ProgressStepView;
use search_types::tasks::Status;
use prometheus::{Encoder, TextEncoder};
use time::OffsetDateTime;

use crate::extractors::authentication::policies::ActionPolicy;
use crate::extractors::authentication::{AuthenticationError, GuardedData};
use crate::routes::create_all_stats;
use crate::routes::indexes::{GetIndexStatsParams, SizeFormat};
use crate::search_queue::SearchQueue;

#[routes::routes(
    routes(
        "" => get(get_metrics),
    ),
    tag = "Stats",
)]
pub struct MetricApi;

/// Get Prometheus metrics
///
/// Return metrics for the engine in Prometheus format. This is an [experimental feature](https://docs.hanzo.ai/docs/search) and must be enabled before use.
#[routes::path(
    security(("Bearer" = ["metrics.get", "metrics.*", "*"])),
    responses(
        (status = 200, description = "The metrics of the instance.", body = String, content_type = "text/plain", example = json!(
            r#"
# HELP index_db_size_bytes DB Size In Bytes
# TYPE index_db_size_bytes gauge
index_db_size_bytes 1130496
# HELP index_batch_running_progress_trace The currently running progress trace
# TYPE index_batch_running_progress_trace gauge
index_batch_running_progress_trace{batch_uid="0",step_name="document"} 0.710618582519409
index_batch_running_progress_trace{batch_uid="0",step_name="extracting word proximity"} 0.2222222222222222
index_batch_running_progress_trace{batch_uid="0",step_name="indexing"} 0.6666666666666666
index_batch_running_progress_trace{batch_uid="0",step_name="processing tasks"} 0
# HELP index_http_requests_total Index HTTP requests total
# TYPE index_http_requests_total counter
index_http_requests_total{method="GET",path="/metrics",status="400"} 1
index_http_requests_total{method="PATCH",path="/experimental-features",status="200"} 1
# HELP index_http_response_time_seconds Hanzo Index HTTP response times
# TYPE index_http_response_time_seconds histogram
index_http_response_time_seconds_bucket{method="GET",path="/metrics",le="0.005"} 0
index_http_response_time_seconds_bucket{method="GET",path="/metrics",le="0.01"} 0
index_http_response_time_seconds_bucket{method="GET",path="/metrics",le="0.025"} 0
index_http_response_time_seconds_bucket{method="GET",path="/metrics",le="0.05"} 0
index_http_response_time_seconds_bucket{method="GET",path="/metrics",le="0.075"} 0
index_http_response_time_seconds_bucket{method="GET",path="/metrics",le="0.1"} 0
index_http_response_time_seconds_bucket{method="GET",path="/metrics",le="0.25"} 0
index_http_response_time_seconds_bucket{method="GET",path="/metrics",le="0.5"} 0
index_http_response_time_seconds_bucket{method="GET",path="/metrics",le="0.75"} 0
index_http_response_time_seconds_bucket{method="GET",path="/metrics",le="1"} 0
index_http_response_time_seconds_bucket{method="GET",path="/metrics",le="2.5"} 0
index_http_response_time_seconds_bucket{method="GET",path="/metrics",le="5"} 0
index_http_response_time_seconds_bucket{method="GET",path="/metrics",le="7.5"} 0
index_http_response_time_seconds_bucket{method="GET",path="/metrics",le="10"} 0
index_http_response_time_seconds_bucket{method="GET",path="/metrics",le="+Inf"} 0
index_http_response_time_seconds_sum{method="GET",path="/metrics"} 0
index_http_response_time_seconds_count{method="GET",path="/metrics"} 0
# HELP index_last_finished_batches_progress_trace_ms The last few batches progress trace in milliseconds
# TYPE index_last_finished_batches_progress_trace_ms gauge
index_last_finished_batches_progress_trace_ms{batch_uid="0",step_name="processing tasks"} 19360
index_last_finished_batches_progress_trace_ms{batch_uid="0",step_name="processing tasks > computing document changes"} 368
index_last_finished_batches_progress_trace_ms{batch_uid="0",step_name="processing tasks > computing document changes > preparing payloads"} 367
index_last_finished_batches_progress_trace_ms{batch_uid="0",step_name="processing tasks > computing document changes > preparing payloads > payload"} 367
index_last_finished_batches_progress_trace_ms{batch_uid="0",step_name="processing tasks > indexing"} 18970
# HELP index_index_count Index Count
# TYPE index_index_count gauge
index_index_count 1
# HELP index_index_docs_count Hanzo Index Index Docs Count
# TYPE index_index_docs_count gauge
index_index_docs_count{index="mieli"} 2
# HELP index_is_indexing Hanzo Index Is Indexing
# TYPE index_is_indexing gauge
index_is_indexing 0
# HELP index_last_update Hanzo Index Last Update
# TYPE index_last_update gauge
index_last_update 1726675964
# HELP index_nb_tasks Hanzo Index Number of tasks
# TYPE index_nb_tasks gauge
index_nb_tasks{kind="indexes",value="mieli"} 39
index_nb_tasks{kind="statuses",value="canceled"} 0
index_nb_tasks{kind="statuses",value="enqueued"} 0
index_nb_tasks{kind="statuses",value="failed"} 4
index_nb_tasks{kind="statuses",value="processing"} 0
index_nb_tasks{kind="statuses",value="succeeded"} 35
index_nb_tasks{kind="types",value="documentAdditionOrUpdate"} 9
index_nb_tasks{kind="types",value="documentDeletion"} 0
index_nb_tasks{kind="types",value="documentEdition"} 0
index_nb_tasks{kind="types",value="dumpCreation"} 0
index_nb_tasks{kind="types",value="indexCreation"} 0
index_nb_tasks{kind="types",value="indexDeletion"} 8
index_nb_tasks{kind="types",value="indexSwap"} 0
index_nb_tasks{kind="types",value="indexUpdate"} 0
index_nb_tasks{kind="types",value="settingsUpdate"} 22
index_nb_tasks{kind="types",value="snapshotCreation"} 0
index_nb_tasks{kind="types",value="taskCancelation"} 0
index_nb_tasks{kind="types",value="taskDeletion"} 0
# HELP index_used_db_size_bytes Used DB Size In Bytes
# TYPE index_used_db_size_bytes gauge
index_used_db_size_bytes 409600
"#
        )),
        (status = 401, description = "The authorization header is missing.", body = ResponseError, content_type = "application/json", example = json!(
            {
                "message": "The Authorization header is missing. It must use the bearer authorization method.",
                "code": "missing_authorization_header",
                "type": "auth",
                "link": "https://docs.hanzo.ai/docs/errors#missing_authorization_header"
            }
        )),
    )
)]
pub async fn get_metrics(
    index_scheduler: GuardedData<ActionPolicy<{ actions::METRICS_GET }>, Data<IndexScheduler>>,
    auth_controller: Data<AuthController>,
    search_queue: web::Data<SearchQueue>,
) -> Result<HttpResponse, ResponseError> {
    index_scheduler.features().check_metrics()?;
    let auth_filters = index_scheduler.filters();
    if !auth_filters.all_indexes_authorized() {
        let mut error = ResponseError::from(AuthenticationError::InvalidToken);
        error
            .message
            .push_str(" The API key for the `/metrics` route must allow access to all indexes.");
        return Err(error);
    }

    let response = create_all_stats(
        (*index_scheduler).clone(),
        auth_controller,
        auth_filters,
        GetIndexStatsParams {
            show_internal_database_sizes: Param(false),
            size_format: Some(SizeFormat::Raw),
        },
    )?;

    let database_size = match response.database_size {
        crate::routes::indexes::Size::Raw(bytes) => bytes as i64,
        crate::routes::indexes::Size::Human(_) => 0,
    };

    let used_database_size = match response.used_database_size {
        crate::routes::indexes::Size::Raw(bytes) => bytes as i64,
        crate::routes::indexes::Size::Human(_) => 0,
    };

    crate::metrics::INDEX_DB_SIZE_BYTES.set(database_size as i64);
    crate::metrics::INDEX_USED_DB_SIZE_BYTES.set(used_database_size as i64);
    crate::metrics::INDEX_INDEX_COUNT.set(response.indexes.len() as i64);

    crate::metrics::INDEX_SEARCH_QUEUE_SIZE.set(search_queue.capacity() as i64);
    crate::metrics::INDEX_SEARCHES_RUNNING.set(search_queue.searches_running() as i64);
    crate::metrics::INDEX_SEARCHES_WAITING_TO_BE_PROCESSED
        .set(search_queue.searches_waiting() as i64);

    for (index, value) in response.indexes.iter() {
        crate::metrics::INDEX_INDEX_DOCS_COUNT
            .with_label_values(&[index])
            .set(value.number_of_documents as i64);
    }

    for (kind, value) in index_scheduler.get_stats()? {
        for (value, count) in value {
            crate::metrics::INDEX_NB_TASKS
                .with_label_values(&[&kind, &value])
                .set(count as i64);
        }
    }

    // Fetch and expose the current progressing step
    crate::metrics::INDEX_BATCH_RUNNING_PROGRESS_TRACE.reset();
    let (batches, _total) = index_scheduler.get_batches_from_authorized_indexes(
        &Query { statuses: Some(vec![Status::Processing]), ..Query::default() },
        auth_filters,
    )?;
    if let Some(batch) = batches.into_iter().next() {
        let batch_uid = batch.uid.to_string();
        if let Some(progress) = batch.progress {
            for ProgressStepView { current_step, finished, total } in progress.steps {
                crate::metrics::INDEX_BATCH_RUNNING_PROGRESS_TRACE
                    .with_label_values(&[batch_uid.as_str(), current_step.as_ref()])
                    // We return the completion ratio of the current step
                    .set(finished as f64 / total as f64);
            }
        }
    }

    crate::metrics::INDEX_LAST_FINISHED_BATCHES_PROGRESS_TRACE_MS.reset();
    let (batches, _total) = index_scheduler.get_batches_from_authorized_indexes(
        // Fetch the finished batches...
        &Query {
            statuses: Some(vec![Status::Succeeded, Status::Failed]),
            limit: Some(1),
            ..Query::default()
        },
        auth_filters,
    )?;
    // ...and get the last batch only.
    if let Some(batch) = batches.into_iter().next() {
        let batch_uid = batch.uid.to_string();
        for (step_name, duration_str) in batch.stats.progress_trace {
            let Some(duration_str) = duration_str.as_str() else { continue };
            match humantime::parse_duration(duration_str) {
                Ok(duration) => {
                    crate::metrics::INDEX_LAST_FINISHED_BATCHES_PROGRESS_TRACE_MS
                        .with_label_values(&[&batch_uid, &step_name])
                        .set(duration.as_millis() as i64);
                }
                Err(e) => tracing::error!("Failed to parse duration: {e}"),
            }
        }
    }

    if let Some(last_update) = response.last_update {
        crate::metrics::INDEX_LAST_UPDATE.set(last_update.unix_timestamp());
    }
    crate::metrics::INDEX_IS_INDEXING.set(index_scheduler.is_task_processing()? as i64);

    let task_queue_latency_seconds = index_scheduler
        .get_tasks_from_authorized_indexes(
            &Query {
                limit: Some(1),
                reverse: Some(true),
                statuses: Some(vec![Status::Enqueued, Status::Processing]),
                ..Query::default()
            },
            auth_filters,
        )?
        .0
        .first()
        .map(|task| (OffsetDateTime::now_utc() - task.enqueued_at).as_seconds_f64())
        .unwrap_or(0.0);
    crate::metrics::INDEX_TASK_QUEUE_LATENCY_SECONDS.set(task_queue_latency_seconds);
    crate::metrics::INDEX_TASK_QUEUE_MAX_SIZE.set(index_scheduler.max_size()? as i64);
    crate::metrics::INDEX_TASK_QUEUE_USED_SIZE.set(index_scheduler.used_size()? as i64);

    crate::metrics::INDEX_TASK_QUEUE_SIZE_UNTIL_STOP_REGISTERING
        .set(index_scheduler.remaining_size_until_task_queue_stop()? as i64);

    let encoder = TextEncoder::new();
    let mut buffer = vec![];
    encoder.encode(&prometheus::gather(), &mut buffer).expect("Failed to encode metrics");

    let response = String::from_utf8(buffer).expect("Failed to convert bytes to string");

    let content_type = ("content-type", prometheus::TEXT_FORMAT);
    Ok(HttpResponse::Ok().insert_header(content_type).body(response))
}
