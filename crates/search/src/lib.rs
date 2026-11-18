// Adding the new module to the crate root
pub mod index_metadata;

#[cfg(test)]
mod tests {
    use super::index_metadata::*;

    #[test]
    fn test_metadata_immutability() {
        let meta1 = IndexMetadata::new("bert-base", "v1", 768, "L2", Some("search: "));
        let meta2 = IndexMetadata::new("bert-base", "v1", 768, "L2", Some("search: "));
        let meta3 = IndexMetadata::new("bert-large", "v1", 1024, "L2", None);

        assert!(meta1.validate_compatibility(&meta2).is_ok());
        assert!(meta1.validate_compatibility(&meta3).is_err());
    }
}
