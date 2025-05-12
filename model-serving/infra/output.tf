output "master_ips" {
  description = "Floating IPs for master nodes"
  value = {
    for k, v in openstack_networking_floatingip_v2.masters_fip : k => v.address
  }
}

output "worker_ips" {
  description = "Floating IPs for worker nodes"
  value = {
    for k, v in openstack_networking_floatingip_v2.workers_fip : k => v.address
  }
}

output "attached_volumes" {
  description = "Device name where volumes are attached"
  value = {
    for k, v in openstack_compute_volume_attach_v2.attach_volumes : k => v.device
  }
}