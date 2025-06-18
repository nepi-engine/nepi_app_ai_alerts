#!/usr/bin/env python
#
# Copyright (c) 2024 Numurus, LLC <https://www.numurus.com>.
#
# This file is part of nepi-engine
# (see https://github.com/nepi-engine).
#
# License: 3-clause BSD, see https://opensource.org/licenses/BSD-3-Clause
#


import os
import time
import sys
import numpy as np
import cv2
import threading
import copy
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from nepi_sdk import nepi_sdk
from nepi_sdk import nepi_utils
from nepi_sdk import nepi_img 

from std_msgs.msg import UInt8, Int32, Float32, Empty, String, Bool, Header
from sensor_msgs.msg import Image
from nepi_interfaces.msg import BoundingBox, BoundingBoxes, ObjectCount

from nepi_interfaces.msg import SaveDataRate, SaveDataStatus

from nepi_app_ai_alerts.msg import AiAlertsStatus, AiAlerts

from nepi_api.messages_if import MsgIF

from nepi_api.connect_node_if import ConnectNodeClassIF
from nepi_api.connect_system_if import ConnectSaveDataIF
from nepi_api.connect_system_if import ConnectStatesIF
from nepi_api.connect_system_if import ConnectTriggersIF
from nepi_api.data_if import ImageIF


APP_NODE_NAME = 'app_ai_alerts'

class ConnectAppAiAlerts:
    msg_if = None
    ready = False
    namespace = '~'

    con_node_if = None

    connected = False
    status_msg = None
    status_connected = False

 
    #######################
    ### IF Initialization
    def __init__(self, 
                namespace = None,
                ):
        ####  IF INIT SETUP ####
        self.class_name = type(self).__name__
        self.base_namespace = nepi_sdk.get_base_namespace()
        self.node_name = nepi_sdk.get_node_name()
        self.node_namespace = nepi_sdk.get_node_namespace()

        ##############################  
        # Create Msg Class
        self.msg_if = MsgIF(log_name = self.class_name)
        self.msg_if.pub_info("Starting IF Initialization Processes")


        ##############################    
        # Initialize Class Variables

        if namespace is None:
            namespace = nepi_sdk.create_namespace(self.base_namespace,APP_NODE_NAME)
        else:
            namespace = namespace
        self.namespace = nepi_sdk.get_full_namespace(namespace)


        ##############################   
        ## Node Setup

        # Configs Config Dict ####################
        self.CFGS_DICT = {
                'namespace': self.namespace
        }


        # Services Config Dict ####################
        self.SRVS_DICT = None


        # Publishers Config Dict ####################
        self.PUBS_DICT = {
            'publish_status': {
                'namespace': self.node_namespace,
                'topic': 'publish_status',
                'msg': Empty,
                'qsize': 10,
                'latch': False
            },
            'enable_app': {
                'namespace': self.node_namespace,
                'topic': 'enable_app',
                'msg': Bool,
                'qsize': 10,
                'latch': False
            },
            'add_all_classes': {
                'namespace': self.node_namespace,
                'topic': 'add_all_classes',
                'msg': Empty,
                'qsize': 10,
                'latch': False
            },
            'remove_all_classes': {
                'namespace': self.node_namespace,
                'topic': 'remove_all_classes',
                'msg': Empty,
                'qsize': 10,
                'latch': False
            },
            'add_class': {
                'namespace': self.node_namespace,
                'topic': 'add_class',
                'msg': String,
                'qsize': 10,
                'latch': False
            },
            'remove_class': {
                'namespace': self.node_namespace,
                'topic': 'remove_class',
                'msg': String,
                'qsize': 10,
                'latch': False
            },
            'set_alert_delay': {
                'namespace': self.node_namespace,
                'topic': 'set_alert_delay',
                'msg': Float32,
                'qsize': 10,
                'latch': False
            },
            'set_clear_delay': {
                'namespace': self.node_namespace,
                'topic': 'set_clear_delay',
                'msg': Float32,
                'qsize': 10,
                'latch': False
            },
            'set_location_str': {
                'namespace': self.node_namespace,
                'topic': 'set_location_str',
                'msg': String,
                'qsize': 10,
                'latch': False
            },
            'set_trigger_delay': {
                'namespace': self.node_namespace,
                'topic': 'set_trigger_delay',
                'msg': Float32,
                'qsize': 10,
                'callback': self.setSnapshotDelayCb, 
                'callback_args': ()
            },
            'enable_snapshot_trigger': {
                'namespace': self.node_namespace,
                'topic': 'enable_snapshot_trigger',
                'msg': Bool,
                'qsize': 10,
                'latch': False            
            },

            'found_object': {
                'namespace': self.node_namespace,
                'topic': '/found_object', #self.ai_mgr_namespace  + "/found_object"
                'msg': ObjectCount,
                'qsize': 1,
                'latch': False
            },
            'bounding_boxes': {
                'namespace': self.node_namespace,
                'topic': '/bounding_boxes', #self.ai_mgr_namespace  + "/bounding_boxes"
                'msg': BoundingBoxes,
                'qsize': 1,
                'latch': False

            }
        }


        # Subscribers Config Dict ####################
        self.SUBS_DICT = {
            'status_sub': {
                'namespace': self.node_namespace,
                'topic': 'status',
                'msg': AiAlertsStatus,
                'qsize': 1,
                'callback': self._statusCb
            },
            'alerts_sub': {
                'namespace': self.node_namespace,
                'topic': 'alerts',
                'msg': AiAlerts,
                'qsize': 1,
                'callback': self._alertsCb
            }                      
        }


        # Create Node Class ####################
        
        self.con_node_if = ConnectNodeClassIF(
                        namespace = self.namespace,
                        configs_dict = self.CFGS_DICT,
                        services_dict = self.SRVS_DICT,
                        pubs_dict = self.PUBS_DICT,
                        subs_dict = self.SUBS_DICT,
                        log_class_name = True,
                        msg_if = self.msg_if
        )

        

        self.con_node_if.wait_for_ready()

        self.con_save_data_if = ConnectSaveDataIF(namespace = self.namespace)

        ##############################
        # Complete Initialization
        self.ready = True
        self.msg_if.pub_info("IF Initialization Complete")
        ###############################
    

    #######################
    # Class Public Methods
    #######################

    def get_ready_state(self):
        return self.ready

    def wait_for_ready(self, timout = float('inf') ):
        success = False
        if self.ready is not None:
            self.msg_if.pub_info("Waiting for connection")
            timer = 0
            time_start = nepi_sdk.get_time()
            while self.ready == False and timer < timeout and not nepi_sdk.is_shutdown():
                nepi_sdk.sleep(.1)
                timer = nepi_sdk.get_time() - time_start
            if self.ready == False:
                self.msg_if.pub_info("Failed to Connect")
            else:
                self.msg_if.pub_info("Connected")
        return self.ready  

    def get_namespace(self):
        return self.namespace

    def check_connection(self):
        return self.connected

    def wait_for_connection(self, timout = float('inf') ):
        if self.con_node_if is not None:
            self.msg_if.pub_info("Waiting for connection")
            timer = 0
            time_start = nepi_sdk.get_time()
            while self.connected == False and timer < timeout and not nepi_sdk.is_shutdown():
                nepi_sdk.sleep(.1)
                timer = nepi_sdk.get_time() - time_start
            if self.connected == False:
                self.msg_if.pub_info("Failed to Connect")
            else:
                self.msg_if.pub_info("Connected")
        return self.connected


    def check_status_connection(self):
        return self.status_connected

    def wait_for_status_connection(self, timout = float('inf') ):
        if self.con_node_if is not None:
            self.msg_if.pub_info("Waiting for status connection")
            timer = 0
            time_start = nepi_sdk.get_time()
            while self.status_connected == False and timer < timeout and not nepi_sdk.is_shutdown():
                nepi_sdk.sleep(.1)
                timer = nepi_sdk.get_time() - time_start
            if self.status_connected == False:
                self.msg_if.pub_info("Failed to connect to status msg")
            else:
                self.msg_if.pub_info("Status Connected")
        return self.status_connected

    def get_status_dict(self):
        img_status_dict = None
        if self.status_msg is not None:
            img_status_dict = nepi_sdk.convert_msg2dict(self.status_msg)
        return self.img_status_dict

    def unregister(self):
        self._unsubscribeTopic()

    def publish_status(self):
        pub_name = 'publish_status'
        msg = Empty()
        self.con_node_if.publish_pub(pub_name,msg)

    def enable_app(self,enable_app):
        pub_name = 'enable_app'
        msg = enable_app
        self.con_node_if.publish_pub(pub_name,msg)  

    def add_all_classes(self):
        pub_name = 'add_all_classes'
        msg = Empty()
        self.con_node_if.publish_pub(pub_name,msg) 

    def reset_controls(self):
        pub_name = 'reset_controls'
        msg = Empty()
        self.con_node_if.publish_pub(pub_name,msg)

    def remove_all_classes(self):
        pub_name = 'remove_all_classes'
        msg = Empty()
        self.con_node_if.publish_pub(pub_name,msg)  

    def add_class(self,add_class):
        pub_name = 'add_class'
        msg = add_class
        self.con_node_if.publish_pub(pub_name,msg)  

    def add_pointcloud(self,pointcloud_ns):
        pub_name = 'add_pointcloud'
        msg = pointcloud_ns    
        self.con_node_if.publish_pub(pub_name,msg)  

    def remove_class(self,remove_class):
        pub_name = 'remove_class'
        msg = remove_class
        self.con_node_if.publish_pub(pub_name,msg)  

    def set_alert_delay(self,alert_delay):
        pub_name = 'set_alert_delay'
        msg = alert_delay
        self.con_node_if.publish_pub(pub_name,msg)  

    def set_clear_delay(self,clear_delay):
        pub_name = 'set_clear_delay'
        msg = clear_delay
        self.con_node_if.publish_pub(pub_name,msg)  

    def set_location_str(self,location_str):
        pub_name = 'set_location_str'
        msg = location_str
        self.con_node_if.publish_pub(pub_name,msg)  

    def set_trigger_delay(self,trigger_delay):
        pub_name = 'set_trigger_delay'
        msg = trigger_delay
        self.con_node_if.publish_pub(pub_name,msg)  

    def enable_snapshot_trigger(self,enable_trigger):
        pub_name = 'enable_snapshot_trigger'
        msg = enable_trigger
        self.con_node_if.publish_pub(pub_name,msg)  

    def found_object(self,found_object):
        pub_name = 'found_object'
        msg = found_object
        self.con_node_if.publish_pub(pub_name,msg)  

    def set_bounding_boxes(self,boxes):
        pub_name = 'bounding_boxes'
        msg = boxes
        self.con_node_if.publish_pub(pub_name,msg) 

    #################
    ## Save Data Functions

    def get_data_products(self):
        data_products = self.con_save_data_if.get_data_products()
        return data_products

    def get_status_dict(self):
        status_dict = self.con_save_data_if.get_status_dict()
        return status_dict

    def save_data_pub(self,enable):
        self.con_save_data_if.save_data_pub(enable)

    def save_data_prefix_pub(self,prefix):
        self.con_save_data_if.save_data_prefix_pub(prefix)

    def save_data_rate_pub(self,rate_hz, data_product = SaveDataRate.ALL_DATA_PRODUCTS):
        self.con_save_data_if.publish_pub(rate_hz, data_product = SaveDataRate.ALL_DATA_PRODUCTS)

    def snapshot_pub(self):
        self.con_save_data_if.publish_pub()

    def reset_pub(self):
        self.con_save_data_if.publish_pub(pub_name,msg)

    def factory_reset_pub(self):
        pub_name = 'factory_reset'
        msg = Empty()
        self.con_save_data_if.publish_pub(pub_name,msg)

    ###############################
    # Class Private Methods
    ###############################
   

    def _unsubscribeTopic(self):
        success = False
        self.connected = False
        if self.con_node_if is not None:
            self.msg_if.pub_warn("Unregistering topic: " + str(self.namespace))
            try:
                self.con_node_if.unregister_class()
                time.sleep(1)
                self.con_node_if = None
                self.namespace = None
                self.status_connected = False 
                self.data_dict = None
                success = True
            except Exception as e:
                self.msg_if.pub_warn("Failed to unregister image:  " + str(e))
        return success


    def _statusCb(self,status_msg):      
        self.status_connected = True
        self.status_msg = status_msg

