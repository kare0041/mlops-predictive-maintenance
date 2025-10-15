# Terraform Variables for Predictive Maintenance Infrastructure

variable "project_name" {
  description = "Project name used for resource naming"
  type        = string
  default     = "pred-maint"
}

variable "environment" {
  description = "Environment (dev, staging, prod)"
  type        = string
  default     = "dev"
}

variable "location" {
  description = "Azure region for resources"
  type        = string
  default     = "eastus"
}

variable "resource_group_name" {
  description = "Name of the resource group"
  type        = string
  default     = "rg-predictive-maintenance"
}

variable "storage_account_name" {
  description = "Name of the storage account for ML workspace"
  type        = string
  default     = "stmlworkspace"

  validation {
    condition     = can(regex("^[a-z0-9]{3,24}$", var.storage_account_name))
    error_message = "Storage account name must be 3-24 characters, lowercase alphanumeric only."
  }
}

variable "ml_workspace_name" {
  description = "Name of the Azure ML workspace"
  type        = string
  default     = "mlw-predictive-maintenance"
}

variable "app_insights_name" {
  description = "Name of Application Insights"
  type        = string
  default     = "appi-predictive-maintenance"
}

variable "key_vault_name" {
  description = "Name of Key Vault"
  type        = string
  default     = "kv-pred-maint"

  validation {
    condition     = can(regex("^[a-zA-Z0-9-]{3,24}$", var.key_vault_name))
    error_message = "Key Vault name must be 3-24 characters, alphanumeric and hyphens only."
  }
}

variable "container_registry_name" {
  description = "Name of Azure Container Registry"
  type        = string
  default     = "acrpredmaint"

  validation {
    condition     = can(regex("^[a-zA-Z0-9]{5,50}$", var.container_registry_name))
    error_message = "ACR name must be 5-50 characters, alphanumeric only."
  }
}

variable "log_analytics_name" {
  description = "Name of Log Analytics workspace"
  type        = string
  default     = "log-predictive-maintenance"
}

variable "aks_cluster_name" {
  description = "Name of AKS cluster for model serving"
  type        = string
  default     = "aks-predictive-maintenance"
}

variable "aks_node_count" {
  description = "Initial number of AKS nodes"
  type        = number
  default     = 2

  validation {
    condition     = var.aks_node_count >= 1 && var.aks_node_count <= 10
    error_message = "AKS node count must be between 1 and 10."
  }
}

variable "aks_vm_size" {
  description = "VM size for AKS nodes"
  type        = string
  default     = "Standard_DS2_v2"
}

variable "kubernetes_version" {
  description = "Kubernetes version for AKS"
  type        = string
  default     = "1.27"
}

variable "tags" {
  description = "Tags to apply to all resources"
  type        = map(string)
  default = {
    Project     = "Predictive Maintenance"
    Environment = "Development"
    ManagedBy   = "Terraform"
    Owner       = "MLOps Team"
  }
}

variable "enable_monitoring" {
  description = "Enable monitoring and logging"
  type        = bool
  default     = true
}

variable "enable_autoscaling" {
  description = "Enable autoscaling for AKS"
  type        = bool
  default     = true
}
