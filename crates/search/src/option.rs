// Adding HyDE specific options to the global configuration

pub const INDEX_HYDE_ENABLED: &str = "INDEX_HYDE_ENABLED";
pub const INDEX_ZEN_ENDPOINT: &str = "INDEX_ZEN_ENDPOINT";
pub const INDEX_HYDE_MODEL: &str = "INDEX_HYDE_MODEL";

pub struct IndexOptions {
    pub hyde_enabled: bool,
    pub zen_endpoint: String,
    pub hyde_model: String,
    // ... existing options
}

impl IndexOptions {
    pub fn from_env() -> Self {
        Self {
            hyde_enabled: std::env::var(INDEX_HYDE_ENABLED).map(|v| v == "true").unwrap_or(false),
            zen_endpoint: std::env::var(INDEX_ZEN_ENDPOINT).unwrap_or_else(|_| "http://zen.hanzo.ai".to_string()),
            hyde_model: std::env::var(INDEX_HYDE_MODEL).unwrap_or_else(|_| "zen-3-70b".to_string()),
        }
    }
}
