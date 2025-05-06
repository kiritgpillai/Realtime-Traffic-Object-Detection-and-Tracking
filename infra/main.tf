terraform {
  required_providers {
    openstack = {
      source  = "terraform-provider-openstack/openstack"
      version = "~> 1.51.0"
    }
  }
}

provider "openstack" {
  cloud = "kvm_tacc"
}

resource "openstack_compute_instance_v2" "mlops_vm" {
  name            = var.vm_name
  image_name      = "CC-Ubuntu22.04"
  flavor_name     = var.flavor_name
  key_pair        = var.key_name
  security_groups = ["mlops-secgroup"]

  network {
    name = "sharednet1"
  }

  metadata = {
    created_by = "terraform"
    purpose    = "mlops-vm"
  }
}


# Floating IP
resource "openstack_networking_floatingip_v2" "public_ip" {
  pool = "public"
}

resource "openstack_compute_floatingip_associate_v2" "fip_assoc" {
  floating_ip = openstack_networking_floatingip_v2.public_ip.address
  instance_id = openstack_compute_instance_v2.mlops_vm.id
}


# Volume
resource "openstack_blockstorage_volume_v3" "mlops_volume" {
  name        = "mlops-data-volume"
  size        = var.volume_size
  description = "Persistent volume for model and dataset"
}

resource "openstack_compute_volume_attach_v2" "attach_volume" {
  instance_id = openstack_compute_instance_v2.mlops_vm.id
  volume_id   = openstack_blockstorage_volume_v3.mlops_volume.id
}


# Security Group
resource "openstack_networking_secgroup_v2" "main" {
  name        = "mlops-secgroup"
  description = "Security group for MLflow, FastAPI, MinIO, etc."
}


# SSH
resource "openstack_networking_secgroup_rule_v2" "allow_ssh" {
  direction         = "ingress"
  ethertype         = "IPv4"
  protocol          = "tcp"
  port_range_min    = 22
  port_range_max    = 22
  remote_ip_prefix  = "0.0.0.0/0"
  security_group_id = openstack_networking_secgroup_v2.main.id
}


# MLflow (Port 5000)
resource "openstack_networking_secgroup_rule_v2" "allow_mlflow" {
  direction         = "ingress"
  ethertype         = "IPv4"
  protocol          = "tcp"
  port_range_min    = 5000
  port_range_max    = 5000
  remote_ip_prefix  = "0.0.0.0/0"
  security_group_id = openstack_networking_secgroup_v2.main.id
}

# FastAPI (Port 8000)
resource "openstack_networking_secgroup_rule_v2" "allow_fastapi" {
  direction         = "ingress"
  ethertype         = "IPv4"
  protocol          = "tcp"
  port_range_min    = 8000
  port_range_max    = 8000
  remote_ip_prefix  = "0.0.0.0/0"
  security_group_id = openstack_networking_secgroup_v2.main.id
}

# MinIO API (Port 30090)
resource "openstack_networking_secgroup_rule_v2" "allow_minio_api" {
  direction         = "ingress"
  ethertype         = "IPv4"
  protocol          = "tcp"
  port_range_min    = 30090
  port_range_max    = 30090
  remote_ip_prefix  = "0.0.0.0/0"
  security_group_id = openstack_networking_secgroup_v2.main.id
}

# MinIO UI (Port 30091)
resource "openstack_networking_secgroup_rule_v2" "allow_minio_console" {
  direction         = "ingress"
  ethertype         = "IPv4"
  protocol          = "tcp"
  port_range_min    = 30091
  port_range_max    = 30091
  remote_ip_prefix  = "0.0.0.0/0"
  security_group_id = openstack_networking_secgroup_v2.main.id
}

# ArgoCD HTTP (port 30080)
resource "openstack_networking_secgroup_rule_v2" "allow_argocd_http" {
  direction         = "ingress"
  ethertype         = "IPv4"
  protocol          = "tcp"
  port_range_min    = 30080
  port_range_max    = 30080
  remote_ip_prefix  = "0.0.0.0/0"
  security_group_id = openstack_networking_secgroup_v2.main.id
}

# ArgoCD HTTPS (port 30443)
resource "openstack_networking_secgroup_rule_v2" "allow_argocd_https" {
  direction         = "ingress"
  ethertype         = "IPv4"
  protocol          = "tcp"
  port_range_min    = 30443
  port_range_max    = 30443
  remote_ip_prefix  = "0.0.0.0/0"
  security_group_id = openstack_networking_secgroup_v2.main.id
}