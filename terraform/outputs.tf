# Terraform Outputs for Predictive Maintenance Infrastructure

output "resource_group_id" {
  description = "ID of the resource group"
  value       = azurerm_resource_group.main.id
}

output "resource_group_location" {
  description = "Location of the resource group"
  value       = azurerm_resource_group.main.location
}

output "ml_workspace_id" {
  description = "ID of the Azure ML workspace"
  value       = azurerm_machine_learning_workspace.main.id
}

output "ml_workspace_endpoint" {
  description = "Discovery URL for the Azure ML workspace"
  value       = azurerm_machine_learning_workspace.main.discovery_url
}

output "storage_account_id" {
  description = "ID of the storage account"
  value       = azurerm_storage_account.mlworkspace.id
}

output "storage_account_primary_key" {
  description = "Primary access key for storage account"
  value       = azurerm_storage_account.mlworkspace.primary_access_key
  sensitive   = true
}

output "key_vault_id" {
  description = "ID of the Key Vault"
  value       = azurerm_key_vault.main.id
}

output "key_vault_uri" {
  description = "URI of the Key Vault"
  value       = azurerm_key_vault.main.vault_uri
}

output "container_registry_id" {
  description = "ID of the container registry"
  value       = azurerm_container_registry.main.id
}

output "container_registry_admin_username" {
  description = "Admin username for container registry"
  value       = azurerm_container_registry.main.admin_username
}

output "container_registry_admin_password" {
  description = "Admin password for container registry"
  value       = azurerm_container_registry.main.admin_password
  sensitive   = true
}

output "aks_id" {
  description = "ID of the AKS cluster"
  value       = azurerm_kubernetes_cluster.main.id
}

output "aks_fqdn" {
  description = "FQDN of the AKS cluster"
  value       = azurerm_kubernetes_cluster.main.fqdn
}

output "aks_kube_config" {
  description = "Kubernetes configuration for AKS"
  value       = azurerm_kubernetes_cluster.main.kube_config_raw
  sensitive   = true
}

output "log_analytics_workspace_id" {
  description = "ID of the Log Analytics workspace"
  value       = azurerm_log_analytics_workspace.main.id
}

output "log_analytics_workspace_key" {
  description = "Primary shared key for Log Analytics"
  value       = azurerm_log_analytics_workspace.main.primary_shared_key
  sensitive   = true
}

output "application_insights_connection_string" {
  description = "Connection string for Application Insights"
  value       = azurerm_application_insights.main.connection_string
  sensitive   = true
}

output "environment_variables" {
  description = "Environment variables for application configuration"
  value = {
    AZURE_SUBSCRIPTION_ID    = data.azurerm_client_config.current.subscription_id
    AZURE_RESOURCE_GROUP     = azurerm_resource_group.main.name
    AZURE_ML_WORKSPACE       = azurerm_machine_learning_workspace.main.name
    AZURE_STORAGE_ACCOUNT    = azurerm_storage_account.mlworkspace.name
    AZURE_KEY_VAULT_NAME     = azurerm_key_vault.main.name
    AZURE_CONTAINER_REGISTRY = azurerm_container_registry.main.login_server
    AZURE_AKS_CLUSTER        = azurerm_kubernetes_cluster.main.name
  }
}
