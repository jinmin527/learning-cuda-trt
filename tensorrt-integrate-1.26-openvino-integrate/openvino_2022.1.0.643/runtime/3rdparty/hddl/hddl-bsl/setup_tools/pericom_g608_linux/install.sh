#!/usr/bin/env bash
# Copyright (C) 2019 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
echo "Copying script to /opt/pericom_g608.sh"
sudo cp pericom_g608.sh /opt/pericom_g608.sh
sudo chmod a+x /opt/pericom_g608.sh

echo "Adding script to /etc/cron.d/pericom_g608_for_hddl"
cron_line="@reboot root /opt/pericom_g608.sh"
sudo echo $cron_line >/etc/cron.d/pericom_g608_for_hddl

sudo /opt/pericom_g608.sh
