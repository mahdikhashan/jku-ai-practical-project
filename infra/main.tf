terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = ">= 4.0"
    }
  }
}

variable "project_id" {}
variable "zone" {
  default = "us-west4-a"
}
variable "region" {
  default = "us-west4"
}
variable "repo_url" {}
variable "script_path" {}

provider "google" {
  project = var.project_id
  region  = var.region
  zone    = var.zone
}

resource "google_compute_instance" "gpu_vm" {
  name         = "temp-gpu-vm"
  machine_type = "n1-standard-4"

  boot_disk {
    initialize_params {
      image = "ubuntu-os-cloud/ubuntu-2204-lts"
    }
  }

  network_interface {
    network = "default"
    access_config {}
  }

  guest_accelerator {
    type  = "nvidia-tesla-t4"
    count = 1
  }

  scheduling {
    on_host_maintenance = "TERMINATE"
    automatic_restart   = true
  }

  metadata = {
    install-nvidia-drivers = "true"
  }

  metadata_startup_script = <<-EOT
    #!/bin/bash
    set -e
    apt-get update
    apt-get install -y git python3 python3-pip
    git clone ${var.repo_url} /tmp/repo
    cd /tmp/repo
    python3 ${var.script_path} || true
  EOT
}
