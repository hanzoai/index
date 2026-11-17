pub enum Decision {
    Retrieve,
    AnswerDirectly,
}

pub struct KaiGatekeeper {
    pub threshold: f32,
}

impl KaiGatekeeper {
    pub fn decide(&self, query: &str) -> Decision {
        // Logic: If query is short or lacks specific entities, answer directly.
        // In a production scenario, this calls a small model or heuristic.
        if query.split_whitespace().count() < 5 {
            Decision::AnswerDirectly
        } else {
            Decision::Retrieve
        }
    }
}
