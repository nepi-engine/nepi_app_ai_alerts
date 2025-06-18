#!/usr/bin/env python
#
# Copyright (c) 2024 Numurus <https://www.numurus.com>.
#
# This file is part of nepi applications (nepi_apps) repo
# (see https://https://github.com/nepi-engine/nepi_apps)
#
# License: nepi applications are licensed under the "Numurus Software License", 
# which can be found at: <https://numurus.com/wp-content/uploads/Numurus-Software-License-Terms.pdf>
#
# Redistributions in source code must retain this top-level comment block.
# Plagiarizing this software to sidestep the license obligations is illegal.
#
# Contact Information:
# ====================
# - mailto:nepi@numurus.com
#
import os
# ROS namespace setup
#NEPI_BASE_NAMESPACE = '/nepi/s2x/'
#os.environ["ROS_NAMESPACE"] = NEPI_BASE_NAMESPACE[0:-1] # remove to run as automation script



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

from nepi_interfaces.srv import ImageClassifierStatusQuery, ImageClassifierStatusQueryRequest

from nepi_app_ai_alerts.msg import AiAlertsStatus, AiAlerts


from nepi_api.connect_node_if import ConnectNodeClassIF
from nepi_api.messages_if import MsgIF
from nepi_api.system_if import SaveDataIF
from nepi_api.system_if import SaveCfgIF
from nepi_api.system_if import TriggersIF
from nepi_api.data_if import ImageIF

# Do this at the end
#from scipy.signal import find_peaks

#########################################
# Node Class
#########################################

class NepiAiAlertsApp(object):
  AI_MANAGER_NODE_NAME = "ai_detector_mgr"

  UDATE_PROCESS_DELAY = 1
  IMG_PUB_PROCESS_DELAY = 0.2

  FACTORY_ALERT_DELAY = 3.0
  FACTORY_CLEAR_DELAY = 2.0
  FACTORY_TRIGGER_DELAY = 10.0

  NONE_CLASSES_DICT = dict()

  STATES_DICT = dict()
  TRIGGERS_DICT = dict()


  data_products = ["alert_image","alert_data"]
  
  current_classifier = "None"
  current_classifier_state = "None"
  classes_list = []
  current_classifier_classes = "[]"

  current_image_topic = "None"
  image_source_topic = ""
  img_width = 0
  img_height = 0


  last_image_topic = "None"
 
  alert_boxes = []
  active_alert = False

  image_if = None
  image_sub = None

  classifier_running = False
  classifier_loading_progress = 0.0
  classifier_threshold = 0.3

  no_object_count = 0

  reset_image_topic = False
  app_enabled = False
  app_msg = "App not enabled"
  img_acquire = False
  img_msg = None
  last_img_msg = None
  img_lock = threading.Lock()


  alert_boxes = []
  alert_boxes_lock = threading.Lock()


  alerts_dict = []
  alerts_dict_lock = threading.Lock()
  
  last_app_enabled = False
  last_trigger_time = None

  classes_selected = False

  last_trigger_time = nepi_sdk.get_msg_time()
  #######################
  ### Node Initialization
  DEFAULT_NODE_NAME = "app_ai_alerts" # Can be overwitten by luanch command
  def __init__(self):
    #### APP NODE INIT SETUP ####
    nepi_sdk.init_node(name= self.DEFAULT_NODE_NAME)
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

    # Message Image to publish when detector not running
    message = "APP NOT ENABLED"
    self.app_ne_img = nepi_img.create_message_image(message)

    message = "WAITING FOR AI DETECTOR TO START"
    self.classifier_nr_img = nepi_img.create_message_image(message)


    message = "WAITING FOR ALERT CLASSES SELECTION"
    self.no_class_img = nepi_img.create_message_image(message)




    ##############################
    ### Setup Node

    # Configs Config Dict ####################
    self.CFGS_DICT = {
        'init_callback': self.initCb,
        'reset_callback': self.resetCb,
        'factory_reset_callback': self.factoryResetCb,
        'init_configs': True,
        'namespace': self.node_namespace
    }


    # Params Config Dict ####################
    self.PARAMS_DICT = {
        'app_enabled': {
            'namespace': self.node_namespace,
            'factory_val': False
        },
        'last_classifier': {
            'namespace': self.node_namespace,
            'factory_val': ""
        },
        'selected_classes': {
            'namespace': self.node_namespace,
            'factory_val': []
        },
        'alert_delay': {
            'namespace': self.node_namespace,
            'factory_val': self.FACTORY_ALERT_DELAY
        },
        'clear_delay': {
            'namespace': self.node_namespace,
            'factory_val': self.FACTORY_CLEAR_DELAY
        },
        'location': {
            'namespace': self.node_namespace,
            'factory_val': ""
        },
        'trigger_delay': {
            'namespace': self.node_namespace,
            'factory_val': self.FACTORY_TRIGGER_DELAY
        },
        'snapshot_trigger_enabled': {
            'namespace': self.node_namespace,
            'factory_val': False
        }

    }


    # Publishers Config Dict ####################
    self.PUBS_DICT = {
        'status_pub': {
            'namespace': self.node_namespace,
            'topic': 'status',
            'msg': AiAlertsStatus,
            'qsize': 1,
            'latch': True
        },
        'alerts_pub': {
            'namespace': self.node_namespace,
            'topic': 'alerts',
            'msg': AiAlerts,
            'qsize': 1,
            'latch': True
        },   
        'snapshot_pub': {
            'namespace': self.node_namespace,
            'topic': 'snapshot_trigger',
            'msg': Empty,
            'qsize': 1,
            'latch': True
        }
        
    }  

    # Subscribers Config Dict ####################
    self.SUBS_DICT = {
        'publish_status': {
            'namespace': self.node_namespace,
            'topic': 'publish_status',
            'msg': Empty,
            'qsize': 10,
            'callback': self.pubStatusCb, 
            'callback_args': ()
        },
        'enable_app': {
            'namespace': self.node_namespace,
            'topic': 'enable_app',
            'msg': Bool,
            'qsize': 10,
            'callback': self.appEnableCb, 
            'callback_args': ()
        },
        'add_all_classes': {
            'namespace': self.node_namespace,
            'topic': 'add_all_classes',
            'msg': Empty,
            'qsize': 10,
            'callback': self.addAllClassesCb, 
            'callback_args': ()
        },
        'remove_all_classes': {
            'namespace': self.node_namespace,
            'topic': 'remove_all_classes',
            'msg': Empty,
            'qsize': 10,
            'callback': self.removeAllClassesCb, 
            'callback_args': ()
        },
        'add_class': {
            'namespace': self.node_namespace,
            'topic': 'add_class',
            'msg': String,
            'qsize': 10,
            'callback': self.addClassCb, 
            'callback_args': ()
        },
        'remove_class': {
            'namespace': self.node_namespace,
            'topic': 'remove_class',
            'msg': String,
            'qsize': 10,
            'callback': self.removeClassCb, 
            'callback_args': ()
        },
        'set_alert_delay': {
            'namespace': self.node_namespace,
            'topic': 'set_alert_delay',
            'msg': Float32,
            'qsize': 10,
            'callback': self.setAlertDelayCb, 
            'callback_args': ()
        },
        'set_clear_delay': {
            'namespace': self.node_namespace,
            'topic': 'set_clear_delay',
            'msg': Float32,
            'qsize': 10,
            'callback': self.setClearDelayCb, 
            'callback_args': ()
        },
        'set_location_str': {
            'namespace': self.node_namespace,
            'topic': 'set_location_str',
            'msg': String,
            'qsize': 10,
            'callback': self.setLocationCb, 
            'callback_args': ()
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
            'callback': self.setSnapshotEnableCb, 
            'callback_args': ()
        },
        'found_object': {
            'namespace': self.node_namespace,
            'topic': '/found_object', #self.ai_mgr_namespace  + "/found_object"
            'msg': ObjectCount,
            'qsize': 1,
            'callback': self.foundObjectCb, 
            'callback_args': ()
        },
        'bounding_boxes': {
            'namespace': self.node_namespace,
            'topic': '/bounding_boxes', #self.ai_mgr_namespace  + "/bounding_boxes"
            'msg': BoundingBoxes,
            'qsize': 1,
            'callback': self.objectDetectedCb, 
            'callback_args': ()
        }
    }
    

    # Create Node Class ####################
    self.node_if = NodeClassIF(
                    configs_dict = self.CFGS_DICT,
                    params_dict = self.PARAMS_DICT,
                    pubs_dict = self.PUBS_DICT,
                    subs_dict = self.SUBS_DICT
    )

    ready = self.node_if.wait_for_ready()


    # Setup Image IF
    self.image_if = ImageIF(namespace = self.node_namespace, topic = 'alert_image')


    # Setup Save Data IF
    factory_data_rates= {}
    for d in self.data_products:
        factory_data_rates[d] = [1.0, 0.0, 100.0] # Default to 1Hz save rate, set last save = 0.0, max rate = 100.0Hz
    self.save_data_if = SaveDataIF(data_products = self.data_products_list, factory_rate_dict = factory_data_rates)

    # Setup States IF
    self.STATES_DICT = {
                    "ai_alert_active": {
                        "name":"ai_alert_active",
                        "node_name": self.node_name,
                        "description": "Current alert detection state",
                        "type":"Bool",
                        "options": [],
                        "value":"False"
                        }
    }
    self.states_if = StatesIF(get_states_dict_function = self.get_states_dict_function)


    # Setup Triggers IF
    self.TRIGGERS_DICT = {
                    "ai_alert_trigger": {
                        "name":"ai_alert_trigger",
                        "namespace": self.namespace,
                        "description": "Triggered on alert detection",
                        "data_str_list":["None"],
                        "time":nepi_utils.get_time()
                        }
    }
    self.triggers_if = TriggersIF(triggers_dict = self.Triggers_Dict)



    ##############################
    # Get AI Manager Service Call
    self.ai_mgr_namespace = self.base_namespace + self.AI_MANAGER_NODE_NAME

    AI_MGR_STATUS_SERVICE_NAME = self.ai_mgr_namespace  + "/img_classifier_status_query"
    self.get_ai_mgr_status_service = self.nepi_sdk.connect_service(AI_MGR_STATUS_SERVICE_NAME, ImageClassifierStatusQuery)
    # Start AI Manager Subscribers
    FOUND_OBJECT_TOPIC = self.ai_mgr_namespace  + "/found_object"
    self.nepi_sdk.create_subscriber(FOUND_OBJECT_TOPIC, ObjectCount, self.foundObjectCb, queue_size = 1)
    BOUNDING_BOXES_TOPIC = self.ai_mgr_namespace  + "/bounding_boxes"
    self.nepi_sdk.create_subscriber(BOUNDING_BOXES_TOPIC, BoundingBoxes, self.objectDetectedCb, queue_size = 1)
    time.sleep(1)


    ##############################
    # Finish Initialization Processes
    self.initCb(do_updates = True)
    self.image_if.publish_cv2_image(self.app_ne_img)

    # Start timed update processes
    nepi_sdk.timer(nepi_sdk.ros_duration(self.UDATE_PROCESS_DELAY), self.updaterCb)
    nepi_sdk.timer(nepi_sdk.ros_duration(self.IMG_PUB_PROCESS_DELAY), self.imagePubCb)

    time.sleep(1)



    self.publish_status()
   
    ## Initiation Complete
    self.msg_if.pub_info(" Initialization Complete")
    # Spin forever (until object is detected)
    nepi_sdk.spin()



  def get_states_dict_function(self):
    return self.STATES_DICT



  #######################
  ### App Config Functions

  def initCb(self,do_updates = False):
      self.msg_if.pub_info(" Setting init values to param values")
      sel_classes = self.node_if.get_param('selected_classes')
      if 'All' in sel_classes:
        self.addAllClasses()
      if do_updates == True:
        self.resetCb(do_updates)


  def resetCb(self):
      self.publish_status()

  def factoryResetCb(self):
    self.last_image_topic = ""
    self.publish_status()

  ###################
  ## Status Publisher
  def publish_status(self):
    status_msg = AiAlertsStatus()

    status_msg.app_enabled = self.node_if.get_param('app_enabled')
    status_msg.app_msg = self.app_msg

    status_msg.location_str = self.node_if.get_param('location')
    status_msg.classifier_running = self.classifier_running


    avail_classes = self.classes_list
    if len(avail_classes) == 0:
      avail_classes = ["None"]
    avail_classes = sorted(avail_classes)
    status_msg.available_classes_list = avail_classes
    selected_classes = self.node_if.get_param('selected_classes')
    sel_classes = []
    for sel_class in selected_classes:
      if sel_class in avail_classes:
        sel_classes.append(sel_class)
    if len(sel_classes) == 0:
      sel_classes = ['None']
    status_msg.selected_classes_list = sel_classes
    status_msg.alert_delay_sec = self.node_if.get_param('alert_delay')
    status_msg.clear_delay_sec = self.node_if.get_param('clear_delay')

    status_msg.trigger_delay_sec = self.node_if.get_param('trigger_delay')
    status_msg.snapshot_trigger_enabled = self.node_if.get_param('snapshot_trigger_enabled')
    self.node_if.publish_pub('status_pub', status_msg)

 
  ## Status Publisher
  def publish_alerts(self,active_alert_boxes):
    if self.active_alert == True:
      alerts_msg = AiAlerts()
      stamp = nepi_sdk.get_msg_time()
      alerts_msg.header.stamp = stamp
      alerts_msg.date_time_str = nepi_sdk.get_datetime_str_from_stamp(stamp)
      alerts_msg.location_str = self.node_if.get_param('location')
      alerts_msg.alert_classes_list = active_alert_boxes
      self.node_if.publish_pub('alerts_pub', alerts_msg)     
    
 

  def updaterCb(self,timer):
    # Save last image topic for next check
    self.last_image_topic = self.current_image_topic
    update_status = False
    app_enabled = self.node_if.get_param("~app_enabled")
    app_msg = ""
    if app_enabled == False:
      app_msg += "App not enabled"
      self.alerts_dict = dict()
      if self.image_sub is not None:
        self.msg_if.pub_warn(" App Disabled, Unsubscribing from Image topic : " + self.last_image_topic)
        self.image_sub.unregister()
        time.sleep(1)
        self.image_sub = None
    elif self.last_app_enabled != app_enabled:
      update_status = True
    self.last_app_enabled = app_enabled


    # Update classifier info
    ai_mgr_status_response = None
    try:
      ai_mgr_status_response = self.get_ai_mgr_status_service()
      #self.msg_if.pub_info(" Got classifier status  " + str(ai_mgr_status_response))
    except Exception as e:
      self.msg_if.pub_warn("Failed to call AI MGR STATUS service" + str(e))
      self.classifier_running = False
      self.node_if.set_param('last_classiier', "")
      app_msg += ", AI Detector not connected"
    if ai_mgr_status_response != None:
      app_msg += ", AI Detector connected"
      #status_str = str(ai_mgr_status_response)
      #self.msg_if.pub_warn(" got ai manager status: " + status_str)
      self.current_image_topic = ai_mgr_status_response.selected_img_topic
      self.current_classifier = ai_mgr_status_response.selected_classifier
      self.current_classifier_state = ai_mgr_status_response.classifier_state
      self.classifier_running = self.current_classifier_state == "Running"
      classes_list = ai_mgr_status_response.selected_classifier_classes
      if classes_list != self.classes_list:
        update_status = True
        self.classes_list = classes_list
        if len(self.classes_list) > 0:
          cmap = plt.get_cmap('viridis')
          color_list = cmap(np.linspace(0, 1, len(self.classes_list))).tolist()
          rgb_list = []
          for color in color_list:
            rgb = []
            for i in range(3):
              rgb.append(int(color[i]*255))
            rgb_list.append(rgb)
          self.class_color_list = rgb_list
      self.classes_list = classes_list
      self.node_if.set_param('last_classiier', self.current_classifier)
      #self.msg_if.pub_warn(" Got image topics last and current: " + self.last_image_topic + " " + self.current_image_topic)

      # Update Image Topic Subscriber
      if self.classifier_running == False:
        app_msg += ", Classifier not running"
        self.alerts_dict = dict()
      else:
        app_msg += ", Classifier running"
        if (self.last_image_topic != self.current_image_topic) or (self.image_sub == None and self.current_image_topic != "None") or self.reset_image_topic == True:
          update_status = True
          self.reset_image_topic = False
          image_topic = nepi_sdk.find_topic(self.current_image_topic)
          if image_topic == "":
            self.msg_if.pub_warn(" Could not find image update topic: " + self.current_image_topic)
          elif app_enabled == True and image_topic != "None":
            self.msg_if.pub_info(" Found detect Image update topic : " + image_topic)
            if self.image_sub != None:
              self.msg_if.pub_warn(" Unsubscribing to Image topic : " + self.last_image_topic)
              self.image_sub.unregister()
              time.sleep(1)
              self.image_sub = None
            self.msg_if.pub_info(" Subscribing to Image topic : " + image_topic)
            self.image_sub = self.nepi_sdk.create_subscriber(image_topic, Image, self.imageCb, queue_size = 1)

        if self.current_image_topic == "None" or self.current_image_topic == "":  # Reset last image topic
          if self.image_sub != None:
            self.msg_if.pub_warn(" Unsubscribing to Image topic : " + self.current_image_topic)
            self.image_sub.unregister()
            time.sleep(1)
            self.image_sub = None
            update_status = True
            time.sleep(1)

    # Print a message image if needed
    sel_classes = self.node_if.get_param('selected_classes')
    classes_selected = False
    for sel_class in sel_classes:
      if sel_class in self.classes_list:
          classes_selected = True
    self.classes_selected = classes_selected
    #self.msg_if.pub_warn("Classes List: " + str(self.classes_list))
    #self.msg_if.pub_warn("Classes List: " + str(sel_classes))
    #self.msg_if.pub_warn("Classes Sel: " + str(self.classes_selected))
    #self.msg_if.pub_warn("" )
    if app_enabled == False:
      #self.msg_if.pub_warn("Publishing Not Enabled image")
      self.image_if.publish_cv2_image(self.app_ne_img)
    elif self.classifier_running == False:
      self.image_if.publish_cv2_image(self.classifier_nr_img)
    elif self.classes_selected == False:
      self.image_if.publish_cv2_image(self.no_class_img)

    self.app_msg = app_msg
    # Publish status if needed
    if update_status == True:
      self.publish_status()
    



  ###################
  ## AI App Callbacks

  def pubStatusCb(self,msg):
    self.publish_status()

  def appEnableCb(self,msg):
    #self.msg_if.pub_info(msg)
    val = msg.data
    self.node_if.set_param('app_enabled', val)
    self.publish_status()

  def addAllClassesCb(self,msg):
    self.addAllClasses()
    self.publish_status()

  def addAllClasses(self):
    ##self.msg_if.pub_info(msg)
    self.node_if.set_param('selected_classes', self.classes_list)


  def removeAllClassesCb(self,msg):
    ##self.msg_if.pub_info(msg)
    self.node_if.set_param('selected_classes',[])
    self.publish_status()

  def addClassCb(self,msg):
    ##self.msg_if.pub_info(msg)
    class_name = msg.data
    if class_name in self.classes_list:
      sel_classes = self.node_if.get_param('selected_classes')
      if class_name not in sel_classes:
        sel_classes.append(class_name)
      self.node_if.set_param('selected_classes', sel_classes)
    self.publish_status()

  def removeClassCb(self,msg):
    ##self.msg_if.pub_info(msg)
    class_name = msg.data
    sel_classes = self.node_if.get_param('selected_classes')
    if class_name in sel_classes:
      sel_classes.remove(class_name)
      self.node_if.set_param('selected_classes', sel_classes)
    self.publish_status()

  def setAlertDelayCb(self,msg):
    self.msg_if.pub_info(msg)
    val = msg.data
    if val >= 0 and val <= 1:
      self.node_if.set_param('alert_delay',val)
    self.publish_status()

  def setClearDelayCb(self,msg):
    self.msg_if.pub_info(msg)
    val = msg.data
    if val >= 0 and val <= 1:
      self.node_if.set_param('clear_delay',val)
    self.publish_status()

  def setLocationCb(self,msg):
    ##self.msg_if.pub_info(msg)
    location_str = msg.data
    self.node_if.set_param('location', location_str)
    self.publish_status()


  def setSnapshotDelayCb(self,msg):
    #self.msg_if.pub_info(msg)
    val = msg.data
    if val > 0 :
      self.node_if.set_param('trigger_delay',val)
    self.publish_status()

        
  def setSnapshotEnableCb(self,msg):
    #self.msg_if.pub_info(msg)
    val = msg.data
    self.node_if.set_param('snapshot_trigger_enabled',val)
    self.publish_status()

         

  #######################
  ### AI Magnager Callbacks


  ### If object(s) detected, save bounding box info to global
  def objectDetectedCb(self,bounding_boxes_msg):
    app_enabled = self.node_if.get_param('app_enabled')
    get_msg_timestamp = bounding_boxes_msg.header.stamp

    if app_enabled == False:
      alert_boxes_acquire = False
      self.alert_boxes = []
      alert_boxes_lock = threading.Lock()
      self.alert_classes = []
    else:
      alert_boxes = []
      sel_classes = self.node_if.get_param('selected_classes')
      for box in bounding_boxes_msg.bounding_boxes:
        if box.Class in sel_classes:
          alert_boxes.append(box)
      if len(alert_boxes) > 0:
        # create save dict
        self.alerts_dict_lock.acquire()
        alerts_dict = self.alerts_dict     
        self.alerts_dict_lock.release()

        for box in alert_boxes:
          box_class = box.Class
          if box_class not in alerts_dict.keys():
            alerts_dict[box_class] = dict()
            alerts_dict[box_class]['first_alert_time'] = get_msg_timestamp
            alerts_dict[box_class]['last_alert_time'] = get_msg_timestamp
          else:
            if 'first_alert_time' not in alerts_dict[box_class].keys():
              alerts_dict[box_class]['first_alert_time'] = get_msg_timestamp
            alerts_dict[box_class]['last_alert_time'] = get_msg_timestamp
            
        self.alert_boxes_lock.acquire()
        self.alert_boxes = alert_boxes      
        self.alert_boxes_lock.release()

        self.alerts_dict_lock.acquire()
        alerts_dict = self.alerts_dict     
        self.alerts_dict_lock.release()

      else:
        self.alert_boxes_lock.acquire()
        self.alert_boxes = []     
        self.alert_boxes_lock.release()



  def imagePubCb(self,timer):
    data_product = 'alert_image'
    has_subscribers = self.image_if.has_subscribers_check()
    #self.msg_if.pub_warn("Checking for subscribers: " + str(has_subscribers))
    saving_is_enabled = self.save_data_if.data_product_saving_enabled(data_product)
    snapshot_enabled = self.save_data_if.data_product_snapshot_enabled(data_product)
    should_save = (saving_is_enabled and self.save_data_if.data_product_should_save(data_product)) or snapshot_enabled
    #self.msg_if.pub_warn("Checking for save_: " + str(should_save))
    app_enabled = self.node_if.get_param('app_enabled')
    
    if app_enabled and self.image_if is not None and self.classifier_running and self.classes_selected:
      if has_subscribers or should_save:
        self.img_lock.acquire()
        img_msg = copy.deepcopy(self.img_msg)
        self.img_msg = None
        self.img_lock.release()
        if img_msg is not None:

          self.alert_boxes_lock.acquire()
          alert_boxes = self.alert_boxes   
          self.alert_boxes_lock.release()


          
          current_image_header = img_msg.header
          get_msg_timestamp = img_msg.header.stamp     
          cv2_img = nepi_img.rosimg_to_cv2img(img_msg)

          #Convert OpenCV image to ROS image
          cv2_shape = cv2_img.shape
          if  cv2_shape[2] == 3:
            encode = 'bgr8'
          else:
            encode = 'mono8'


          for box in alert_boxes:
            #self.msg_if.pub_warn(" Box: " + str(box))
            class_name = box.Class
            [xmin,xmax,ymin,ymax] = [box.xmin,box.xmax,box.ymin,box.ymax]
            start_point = (xmin, ymin)
            end_point = (xmax, ymax)
            class_name = class_name
            class_color = (0,0,255)
            line_thickness = 2
            cv2.rectangle(cv2_img, start_point, end_point, class_color, thickness=line_thickness)

          self.image_if.publish_cv2_image(cv2_img, timestamp = get_msg_timestamp, encoding=encode)
          # Save Data if 
          if should_save:
            self.save_data_if.save_img2file(data_product,cv2_img,get_msg_timestamp,save_check = False)

  def imageCb(self,image_msg):   
      #self.msg_if.pub_warn("Got image msg: ") 
      self.img_lock.acquire()
      self.img_msg = copy.deepcopy(self.last_img_msg)
      self.img_lock.release()
      self.last_img_msg = copy.deepcopy(image_msg)



  ### Monitor Output of AI model to clear detection status
  def foundObjectCb(self,found_obj_msg):
    app_enabled = self.node_if.get_param('app_enabled')
    get_msg_timestamp = found_obj_msg.header.stamp
    #Clean Up Detection and Alert data
    if found_obj_msg.count == 0:
      self.alert_boxes_lock.acquire()
      self.alert_boxes = []     
      self.alert_boxes_lock.release()

    # Update alert dict and status info
    self.alerts_dict_lock.acquire()
    alerts_dict = self.alerts_dict     
    self.alerts_dict_lock.release()

    # Purge old alerts
    clear_delay = self.node_if.get_param('clear_delay')
    purge_alert_list = []
    for key in alerts_dict.keys():
      last_alert_time =(get_msg_timestamp.to_sec() - alerts_dict[key]['last_alert_time'].to_sec())
      if last_alert_time > clear_delay:
        purge_alert_list.append(key)
    for alert in purge_alert_list:
      del alerts_dict[alert]
    self.alerts_dict_lock.acquire()
    self.alerts_dict = alerts_dict
    self.alerts_dict_lock.release()

    # Check current alert trigger time
    active_alert = False
    active_alert_classes = []
    alert_delay = self.node_if.get_param('alert_delay')
    for key in alerts_dict.keys():
      first_alert_time =(get_msg_timestamp.to_sec() - alerts_dict[key]['first_alert_time'].to_sec())
      if first_alert_time > alert_delay:
        active_alert = True
        if key not in active_alert_classes:
          active_alert_classes.append(key)
    self.alert_classes = active_alert_classes
    self.active_alert = active_alert

    if len(active_alert_classes) > 0:
      self.publish_alerts(active_alert_classes)
      alerts_save_dict = dict()
      alerts_save_dict['timestamp'] = nepi_sdk.get_datetime_str_from_stamp(get_msg_timestamp)
      alerts_save_dict['location'] = self.node_if.get_param('location')
      alerts_save_dict['alert_classes_list'] = active_alert_classes
      self.save_data_if.save_dict2file('alert_data',alerts_save_dict,get_msg_timestamp,save_check = True)


    # Do stuff on active alert state
    if active_alert == True:
        trigger_delay = self.node_if.get_param('trigger_delay')
        trigger_time = (get_msg_timestamp.to_sec() - self.last_trigger_time.to_sec())
        if (trigger_time > trigger_delay):
          self.last_trigger_time = get_msg_timestamp
          self.node_if.publish_pub('alert_trigger_pub', Empty())
          snapshot_trigger_enabled = self.node_if.get_param('snapshot_trigger_enabled')
          if snapshot_trigger_enabled:
            self.node_if.publish_pub('snapshot_pub', Empty())

        if 'ai_alert_state' in self.STATES_DICT.keys():
          self.STATES_DICT['ai_alert_state']['value'] = str(self.active_alert)


        if 'ai_alert_trigger' in self.Triggers_Dict.keys():
          trigger_dict = self.Triggers_Dict['ai_alert_trigger']
          trigger_dict['time']=trigger_time
          self.triggers_if.publish_trigger(trigger_dict)


        # Publish and save active alert boxes
        self.alert_boxes_lock.acquire()
        alert_boxes = self.alert_boxes    
        self.alert_boxes_lock.release()

    else:
      self.last_trigger_time = get_msg_timestamp

  
               
    
  #######################
  # Node Cleanup Function
  
  def cleanup_actions(self):
    self.msg_if.pub_info(" Shutting down: Executing script cleanup actions")


#########################################
# Main
#########################################
if __name__ == '__main__':
  NepiAiAlertsApp()







