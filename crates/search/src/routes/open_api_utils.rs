use serde::Serialize;
use utoipa::openapi::security::{HttpAuthScheme, HttpBuilder, SecurityScheme};

#[derive(Debug, Serialize)]
pub struct OpenApiAuth;

impl utoipa::Modify for OpenApiAuth {
    fn modify(&self, openapi: &mut utoipa::openapi::OpenApi) {
        if let Some(schema) = openapi.components.as_mut() {
            schema.add_security_scheme(
                "Bearer",
                SecurityScheme::Http(
                    HttpBuilder::new()
                        .scheme(HttpAuthScheme::Bearer)
                        .bearer_format("Uuidv4, string or JWT")
                        .description(Some(
"An API key is a token that you provide when making API calls. Read more about [how to secure your project](https://docs.hanzo.ai/docs/search).\n\nInclude the API key to the `Authorization` header, for instance:\n```bash\n-H \'Authorization: Bearer 6436fc5237b0d6e0d64253fbaac21d135012ecf1\'\n```\n\nIf you use a SDK, ensure you instantiate the client with the API key.\n```js\nconst client = new Index({\n  host: 'INDEX_URL',\n  apiKey: '6436fc5237b0d6e0d64253fbaac21d135012ecf1'\n});\n```"))
                        .build(),
                ),
            );
        }
    }
}
