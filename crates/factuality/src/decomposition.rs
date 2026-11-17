// Logic to decompose generated assertions into atomic claims
pub fn decompose_assertion(assertion: &str) -> Vec<String> {
    // In a production scenario, this calls a specialized Zen prompt
    // For this implementation, we split by sentence/clause as a baseline
    assertion.split_inclusive(['.', '!', '?']).map(|s| s.trim().to_string()).collect()
}
