terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = ">= 4.0"
    }
    tls = {
      source  = "hashicorp/tls"
      version = ">= 4.0"
    }
    local = {
      source  = "hashicorp/local"
      version = ">= 2.0"
    }
  }
}

variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "zone" {
  description = "GCP Zone"
  type        = string
  default     = "asia-southeast1-a"
}

variable "region" {
  description = "GCP Region"
  type        = string
  default     = "asia-southeast1"
}

variable "instance_name" {
  description = "Name of the VM instance"
  type        = string
  default     = "temp-gpu-vm"
}

variable "machine_type" {
  description = "Machine type for the VM"
  type        = string
  default     = "g2-standard-4"
}

variable "boot_disk_image" {
  description = "Boot disk image"
  type        = string
  default     = "deeplearning-platform-release/pytorch-2-7-cu128-ubuntu-2404-nvidia-570-v20251209"
}

variable "network" {
  description = "Network name"
  type        = string
  default     = "default"
}

variable "gpu_type" {
  description = "GPU accelerator type"
  type        = string
  default     = "nvidia-l4"
}

variable "gpu_count" {
  description = "Number of GPUs"
  type        = number
  default     = 1
}

variable "on_host_maintenance" {
  description = "Host maintenance policy"
  type        = string
  default     = "TERMINATE"
}

variable "automatic_restart" {
  description = "Whether to automatically restart VM"
  type        = bool
  default     = true
}

variable "install_nvidia_drivers" {
  description = "Whether to install NVIDIA drivers"
  type        = string
  default     = "true"
}

provider "google" {
  project = var.project_id
  region  = var.region
  zone    = var.zone
}

resource "tls_private_key" "ssh_key" {
  algorithm = "RSA"
  rsa_bits  = 4096
}

resource "local_file" "private_key" {
  content         = tls_private_key.ssh_key.private_key_pem
  filename        = "${path.module}/ssh_key"
  file_permission = "0600"
}

resource "local_file" "public_key" {
  content         = tls_private_key.ssh_key.public_key_openssh
  filename        = "${path.module}/ssh_key.pub"
  file_permission = "0644"
}

resource "google_compute_instance" "gpu_vm" {
  name         = var.instance_name
  machine_type = var.machine_type

  boot_disk {
    initialize_params {
      image = var.boot_disk_image
    }
  }

  network_interface {
    network = var.network
    access_config {}
  }

  guest_accelerator {
    type  = var.gpu_type
    count = var.gpu_count
  }

  scheduling {
    on_host_maintenance = var.on_host_maintenance
    automatic_restart   = var.automatic_restart
  }

  metadata = {
    install-nvidia-drivers = var.install_nvidia_drivers
    ssh-keys              = "ubuntu:${tls_private_key.ssh_key.public_key_openssh}"
  }
}

output "instance_name" {
  description = "Name of the created instance"
  value       = google_compute_instance.gpu_vm.name
}

output "instance_external_ip" {
  description = "External IP address of the instance"
  value       = google_compute_instance.gpu_vm.network_interface[0].access_config[0].nat_ip
}

output "instance_internal_ip" {
  description = "Internal IP address of the instance"
  value       = google_compute_instance.gpu_vm.network_interface[0].network_ip
}

output "ssh_command" {
  description = "SSH command to connect to the instance"
  value       = "ssh -i ssh_key ubuntu@${google_compute_instance.gpu_vm.network_interface[0].access_config[0].nat_ip}"
}