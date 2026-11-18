use serde::{Deserialize, Serialize};
use anyhow::Result;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HydeConfig {
    pub enabled: bool,
    pub model: String,
    pub prompt_template: String,
    pub max_tokens: usize,
}

impl Default for HydeConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            model: "zen-latest".to_string(),
            prompt_template: "Write a short passage to answer this search query: {query}".to_string(),
            max_tokens: 256,
        }
    }
}

pub struct HydeGenerator {
    config: HydeConfig,
    zen_endpoint: String,
}

impl HydeGenerator {
    pub fn new(config: HydeConfig, zen_endpoint: String) -> Self {
        Self { config, zen_endpoint }
    }

    /// Generates a hypothetical document using the Zen model.
    /// This follows the Enso routing logic to reach the Zen generation service.
    pub async fn generate_hypothetical_doc(&self, query: &str) -> Result<String> {
        if !self.config.enabled {
            return Ok(query.to_string());
        }

        let prompt = self.config.prompt_template.replace("{query}", query);

        // In the Hanzo stack, Zen is accessed via Enso routing.
        // This simulates the internal RPC/HTTP call to the Zen generation service.
        let client = reqwest::Client::new();
        let response = client
            .post(&format!("{}/v1/chat/completions", self.zen_endpoint))
            .json(&serde_json::json!({
                "model": self.config.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": self.config.max_tokens,
                "temperature": 0.3,
            }))
            .send()
            .await?;

        let result: serde_json::Value = response.json().await?;
        let synthetic_doc = result["choices"][0]["message"]["content"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("Failed to extract content from Zen response"))?
            .to_string();

        Ok(synthetic_doc)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_hyde_prompt_construction() {
        let config = HydeConfig {
            enabled: true,
            ..Default::default()
        };
        let query = "How does vector indexing work?";
        let prompt = config.prompt_template.replace("{query}", query);
        assert!(prompt.contains("How does vector indexing work?"));
    }
}
