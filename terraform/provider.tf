# Azure Provider Configuration

provider "azurerm" {
  features {
    key_vault {
      purge_soft_delete_on_destroy    = true
      recover_soft_deleted_key_vaults = true
    }

    resource_group {
      prevent_deletion_if_contains_resources = false
    }

    log_analytics_workspace {
      permanently_delete_on_destroy = true
    }
  }
}

provider "azuread" {
  # Azure AD provider for identity management
}
