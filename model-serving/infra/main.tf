locals {
  master_names = [for i in range(var.master_count) : "k8s-master-${i}"]
  worker_names = [for i in range(var.worker_count) : "k8s-worker-${i}"]
}

terraform {
  required_providers {
    openstack = {
      source  = "terraform-provider-openstack/openstack"
      version = "~> 1.51.0"
    }
  }
}

# Security Group
resource "openstack_networking_secgroup_v2" "main" {
  name        = "k8s-secgroup"
  description = "Security group for Kubernetes cluster nodes"
}

# SSH
resource "openstack_networking_secgroup_rule_v2" "ssh" {
  direction         = "ingress"
  ethertype         = "IPv4"
  protocol          = "tcp"
  port_range_min    = 22
  port_range_max    = 22
  remote_ip_prefix  = "0.0.0.0/0"
  security_group_id = openstack_networking_secgroup_v2.main.id
}

# Allow all Kubernetes traffic
resource "openstack_networking_secgroup_rule_v2" "k8s_ports" {
  direction         = "ingress"
  ethertype         = "IPv4"
  protocol          = "tcp"
  port_range_min    = 1
  port_range_max    = 65535
  remote_ip_prefix  = "0.0.0.0/0"
  security_group_id = openstack_networking_secgroup_v2.main.id
}


provider "openstack" {
  cloud = "kvm_tacc"
}


# Create master nodes
resource "openstack_compute_instance_v2" "masters" {
  for_each        = toset(local.master_names)
  name            = each.value
  image_name      = "CC-Ubuntu22.04"
  flavor_name     = var.flavor_name
  key_pair        = var.key_name
  security_groups = [openstack_networking_secgroup_v2.main.name]

  network {
    name = "sharednet1"
  }

  metadata = {
    created_by = "terraform"
    purpose    = "k8s-master"
  }
}

# Create worker nodes
resource "openstack_compute_instance_v2" "workers" {
  for_each        = toset(local.worker_names)
  name            = each.value
  image_name      = "CC-Ubuntu22.04"
  flavor_name     = var.flavor_name
  key_pair        = var.key_name
  security_groups = [openstack_networking_secgroup_v2.main.name]

  network {
    name = "sharednet1"
  }

  metadata = {
    created_by = "terraform"
    purpose    = "k8s-worker"
  }
}

# Floating IPs for all VMs
resource "openstack_networking_floatingip_v2" "masters_fip" {
  for_each = openstack_compute_instance_v2.masters
  pool     = "public"
}

resource "openstack_networking_floatingip_v2" "workers_fip" {
  for_each = openstack_compute_instance_v2.workers
  pool     = "public"
}

resource "openstack_compute_floatingip_associate_v2" "masters_assoc" {
  for_each    = openstack_compute_instance_v2.masters
  floating_ip = openstack_networking_floatingip_v2.masters_fip[each.key].address
  instance_id = each.value.id
}

resource "openstack_compute_floatingip_associate_v2" "workers_assoc" {
  for_each    = openstack_compute_instance_v2.workers
  floating_ip = openstack_networking_floatingip_v2.workers_fip[each.key].address
  instance_id = each.value.id
}

# One volume per VM
resource "openstack_blockstorage_volume_v3" "volumes" {
  for_each    = merge(openstack_compute_instance_v2.masters, openstack_compute_instance_v2.workers)
  name        = "${each.key}-volume"
  size        = var.volume_size
  description = "Volume for ${each.key}"
}

resource "openstack_compute_volume_attach_v2" "attach_volumes" {
  for_each   = openstack_blockstorage_volume_v3.volumes
  instance_id = (merge(openstack_compute_instance_v2.masters, openstack_compute_instance_v2.workers))[each.key].id
  volume_id   = each.value.id
}

# VXLAN
resource "openstack_networking_secgroup_rule_v2" "flannel_vxlan" {
  direction         = "ingress"
  ethertype         = "IPv4"
  protocol          = "udp"
  port_range_min    = 8472
  port_range_max    = 8472
  remote_ip_prefix  = "0.0.0.0/0"
  security_group_id = openstack_networking_secgroup_v2.main.id
}
