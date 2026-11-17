use crate::kai::{KaiGatekeeper, Decision};

pub async fn process_query(query: &str) -> String {
    let gatekeeper = KaiGatekeeper { threshold: 0.7 };
    
    match gatekeeper.decide(query) {
        Decision::Retrieve => {
            let context = perform_retrieval(query).await;
            generate_response(query, Some(context)).await
        },
        Decision::AnswerDirectly => {
            generate_response(query, None).await
        }
    }
}
