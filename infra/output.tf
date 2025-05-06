output "instance_ip" {
  description = "Floating IP address of the instance"
  value       = openstack_networking_floatingip_v2.public_ip.address
}

output "volume_id" {
  description = "ID of the block volume"
  value       = openstack_blockstorage_volume_v3.mlops_volume.id
}

