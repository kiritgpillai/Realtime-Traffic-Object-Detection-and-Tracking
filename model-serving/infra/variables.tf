variable "key_name" {
  description = "SSH key name"
  type        = string
}

variable "flavor_name" {
  description = "Flavor name for VM"
  type        = string
}

variable "master_count" {
  description = "Number of Kubernetes master nodes"
  type        = number
  default     = 1
}

variable "worker_count" {
  description = "Number of Kubernetes worker nodes"
  type        = number
  default     = 2
}

variable "volume_size" {
  description = "Size of the attached volume in GB"
  type        = number
  default     = 100
}
