variable "key_name" {
  description = "SSH key name"
  type        = string
}

variable "flavor_name" {
  description = "Flavor name for VM"
  type        = string
}

variable "vm_name" {
  description = "Name of the compute instance"
  type        = string
}

variable "volume_size" {
  description = "Size of the attached volume in GB"
  type        = number
}

variable "bucket_name" {
  description = "Name of the object storage bucket"
  type        = string
}
