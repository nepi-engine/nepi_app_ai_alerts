/*
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
 */
import React, { Component } from "react"
import { observer, inject } from "mobx-react"

import Section from "./Section"
//import EnableAdjustment from "./EnableAdjustment"
import Button, { ButtonMenu } from "./Button"
import Label from "./Label"
import { Column, Columns } from "./Columns"
import Input from "./Input"
import Select, { Option } from "./Select"
import Styles from "./Styles"
import Toggle from "react-toggle"
import BooleanIndicator from "./BooleanIndicator"


import ImageViewer from "./Nepi_IF_ImageViewer"
import NepiIFSaveData from "./Nepi_IF_SaveData"


import {onUpdateSetStateValue, onEnterSendFloatValue} from "./Utilities"

@inject("ros")
@observer

// Component that contains the  Pointcloud App Viewer Controls
class AppAiAlerts extends Component {
  constructor(props) {
    super(props)

    // these states track the values through  Status messages
    this.state = {

      appName: "app_ai_alerts",
      appNamespace: null,

      app_enabled: false,
      app_msg: "Connecting",
      image_name: "alert_image",

      location_str: "",

      classifier_running: false,

      image_topic: null,
      
      viewableClasses: false,
      available_classes_list: [],
      selected_classes_list:[],

      alert_delay_sec : null,
      clear_delay_sec : null,

      trigger_delay_sec: null,
      trigger_count: 0,
      snapshot_enabled: null,
      event_enabled: null,

      alert_state: false,
        


      statusListener: null,
      alertListener: null,
      triggerListener: null,

      connected: false,
      needs_update: true


    }
  
    this.getAppNamespace = this.getAppNamespace.bind(this)
    this.statusListener = this.statusListener.bind(this)
    this.updateStatusListener = this.updateStatusListener.bind(this)

    this.alertListener = this.alertListener.bind(this)
    this.updateAlertListener = this.updateAlertListener.bind(this)

    this.triggerListener = this.triggerListener.bind(this)
    this.updateTriggerListener = this.updateTriggerListener.bind(this)
    
    this.onToggleClassSelection = this.onToggleClassSelection.bind(this)
    this.getClassOptions = this.getClassOptions.bind(this)
    this.toggleViewableClassesTopics = this.toggleViewableClassesTopics.bind(this)

  }

  getAppNamespace(){
    const { namespacePrefix, deviceId} = this.props.ros
    var appNamespace = null
    if (namespacePrefix !== null && deviceId !== null){
      appNamespace = "/" + namespacePrefix + "/" + deviceId + "/" + this.state.appName
      if (this.state.connected === false){
        const pub_status_topic = appNamespace + "/publish_status"
        this.props.ros.sendTriggerMsg(pub_status_topic)
      }
    }
    return appNamespace
  }

  // Callback for handling ROS Status messages
  statusListener(message) {
    this.setState({

    app_enabled: message.app_enabled,
    app_msg: message.app_msg,

    location_str: message.location_str,

    image_topic: message.image_topic,

    classifier_running: message.classifier_running,

    available_classes_list: message.available_classes,
    selected_classes_list: message.selected_classes,

    alert_delay_sec : message.alert_delay_sec ,
    clear_delay_sec : message.clear_delay_sec ,
  
    trigger_delay_sec: message.trigger_delay_sec,
    snapshot_enabled: message.snapshot_trigger_enabled,
    event_enabled: message.event_trigger_enabled,

    })

    this.setState({
      connected: true
    })

  }

  // Callback for handling ROS Status messages
  alertListener(message) {
    this.setState({
      alert_state: message.data
    })
  }

  // Callback for handling ROS Status messages
  triggerListener(message) {
    const count = this.state.trigger_count + 1
    this.setState({
      trigger_count: count
    })
  }

    // Function for configuring and subscribing to Status
    updateStatusListener() {
      const appNamespace = this.getAppNamespace()
      const statusNamespace = appNamespace + '/status'
      if (this.state.statusListener) {
        this.state.statusListener.unsubscribe()
      }
      var statusListener = this.props.ros.setupStatusListener(
            statusNamespace,
            "nepi_app_ai_alerts/AiAlertsStatus",
            this.statusListener
          )
      this.setState({ 
        statusListener: statusListener,
      })
      this.render()
    }

    // Function for configuring and subscribing to Status
    updateAlertListener() {
      const appNamespace = this.getAppNamespace()
      const statusNamespace = appNamespace + '/alert_state'
      if (this.state.alertListener) {
        this.state.alertListener.unsubscribe()
      }
      var alertListener = this.props.ros.setupStatusListener(
            statusNamespace,
            "std_msgs/Bool",
            this.alertListener
          )
      this.setState({ 
        alertListener: alertListener,
      })
      this.render()
    }

    // Function for configuring and subscribing to Status
    updateTriggerListener() {
      const appNamespace = this.getAppNamespace()
      const statusNamespace = appNamespace + '/alert_trigger'
      if (this.state.triggerListener) {
        this.state.triggerListener.unsubscribe()
      }
      var triggerListener = this.props.ros.setupStatusListener(
            statusNamespace,
            "std_msgs/Empty",
            this.triggerListener
          )
      this.setState({ 
        triggerListener: triggerListener,
      })
      this.render()
    }



  // Lifecycle method called when compnent updates.
  // Used to track changes in the topic
  componentDidUpdate(prevProps, prevState, snapshot) {
    const namespace = this.getAppNamespace()
    const namespace_updated = (prevState.appNamespace !== namespace && namespace !== null)
    if (namespace_updated) {
      if (namespace.indexOf('null') === -1){
        this.setState({appNamespace: namespace})
        this.updateStatusListener()
        this.updateAlertListener()
        this.updateTriggerListener()
      } 
    }
  }


  // Lifecycle method called just before the component umounts.
  // Used to unsubscribe to Status message
  componentWillUnmount() {
    if (this.state.statusListener) {
      this.state.statusListener.unsubscribe()
      this.state.alertListener.unsubscribe()
      this.state.triggerListener.unsubscribe()
    }
  }


  // Function for creating image topic options.
  getClassOptions() {
  const classesList = this.state.available_classes_list
  var items = []
  items.push(<Option>{"None"}</Option>)
  items.push(<Option>{"All"}</Option>)
  if (classesList.length > 0 ){
    for (var i = 0; i < classesList.length; i++) {
        if (classesList[i] !== 'None'){
          items.push(<Option value={classesList[i]}>{classesList[i]}</Option>)
        }
    }
  }
  return items
  }


  toggleViewableClassesTopics() {
    const set = !this.state.viewableClasses
    this.setState({viewableClasses: set})
  }


  onToggleClassSelection(event){
    const {sendTriggerMsg, sendStringMsg} = this.props.ros
    const appNamespace = this.getAppNamespace()
    const classSelection = event.target.value
    const selectedClassesList = this.state.selected_classes_list
    const addAllNamespace = appNamespace + "/add_all_alert_classes"
    const removeAllNamespace = appNamespace + "/remove_all_alert_classes"
    const addNamespace = appNamespace + "/add_alert_class"
    const removeNamespace = appNamespace + "/remove_alert_class"
    if (appNamespace){
      if (classSelection === "None"){
          sendTriggerMsg(removeAllNamespace)
      }
      else if (classSelection === "All"){
        sendTriggerMsg(addAllNamespace)
    }
      else if (selectedClassesList.indexOf(classSelection) !== -1){
        sendStringMsg(removeNamespace,classSelection)
      }
      else {
        sendStringMsg(addNamespace,classSelection)
      }
    }
  }


 

  renderApp() {
    const {sendBoolMsg, sendTriggerMsg,} = this.props.ros
    const classOptions = this.getClassOptions()
    const selectedClasses = this.state.selected_classes_list
    const classes_sel = selectedClasses[0] !== "" && selectedClasses[0] !== "None"
    const classifier_running = this.state.classifier_running
    const connected = this.state.connected === true
    const appNamespace = this.getAppNamespace()


    return (
      <Section title={"AI Alerts App"}>

        <Columns>
        <Column>

        <Columns>
        <Column>

        <div hidden={(connected === true)}>

      <pre style={{ height: "40px", overflowY: "auto" ,fontWeight: 'bold' , color: Styles.vars.colors.Green, textAlign: "left" }}>
          {"Loading or Refresh Page"}
        </pre>

      </div>

      <div hidden={(connected === false)}>

        <Label title="Enable App">
            <Toggle
            checked={this.state.app_enabled===true}
            onClick={() => sendBoolMsg(appNamespace + "/enable_app",!this.state.app_enabled)}>
            </Toggle>
      </Label>

      </div>


          </Column>
        <Column>


        </Column>
      </Columns>


      <div hidden={(connected !== true || this.state.app_enabled !== true)}>

          <Columns>
          <Column>


      <Label title={"AI Detection Running"}>
        <BooleanIndicator value={classifier_running} />
      </Label>

      <Label title={"Alert Classes Selected"}>
        <BooleanIndicator value={classes_sel} />
      </Label>

            </Column>
          <Column>

          <Label title={"Alert State"}>
        <BooleanIndicator value={this.state.alert_state} />
          </Label>
  
          </Column>
        </Columns>

   

        <div style={{ borderTop: "1px solid #ffffff", marginTop: Styles.vars.spacing.medium, marginBottom: Styles.vars.spacing.xs }}/>

       
        <label style={{fontWeight: 'bold'}} align={"left"} textAlign={"left"}>
          {"App Settings"}
         </label>


        <Columns>
          <Column>


          <Label title={"Set Location"}>
                <Input
                  value={this.state.location_str !== "" ? this.state.location_str : "Not Set"}
                  id="Location"
                  onChange= {(event) => onUpdateSetStateValue.bind(this)(event,"location_str")}
                  onKeyDown= {(event) => onEnterSendFloatValue.bind(this)(event,appNamespace + "/set_location_str")}
                  style={{ width: "80%" }}
                />
              </Label>

         <Label title="Select Alert Classes"> </Label>

                    <div onClick={this.toggleViewableClassesTopics} style={{backgroundColor: Styles.vars.colors.grey0}}>
                      <Select style={{width: "10px"}}/>
                    </div>
                    <div hidden={this.state.viewableClasses === false}>
                    {classOptions.map((Class) =>
                    <div onClick={this.onToggleClassSelection}
                      style={{
                        textAlign: "center",
                        padding: `${Styles.vars.spacing.xs}`,
                        color: Styles.vars.colors.black,
                        backgroundColor: (selectedClasses.includes(Class.props.value))? Styles.vars.colors.blue : Styles.vars.colors.grey0,
                        cursor: "pointer",
                        }}>
                        <body class_name ={Class} style={{color: Styles.vars.colors.black}}>{Class}</body>
                    </div>
                    )}
                    </div>

          </Column>
          <Column>


          <Label title={"Alert Delay (sec)"}>
                <Input
                  value={this.state.alert_delay_sec}
                  id="trigger_time"
                  onChange= {(event) => onUpdateSetStateValue.bind(this)(event,"alert_delay_sec")}
                  onKeyDown= {(event) => onEnterSendFloatValue.bind(this)(event,appNamespace + "/set_alert_delay")}
                  style={{ width: "80%" }}
                />
              </Label>


              <Label title={"Clear Delay (sec)"}>
                <Input
                  value={this.state.clear_delay_sec}
                  id="clear _time"
                  onChange= {(event) => onUpdateSetStateValue.bind(this)(event,"clear_delay_sec")}
                  onKeyDown= {(event) => onEnterSendFloatValue.bind(this)(event,appNamespace + "/set_clear_delay")}
                  style={{ width: "80%" }}
                />
              </Label>

      <div style={{ borderTop: "1px solid #ffffff", marginTop: Styles.vars.spacing.medium, marginBottom: Styles.vars.spacing.xs }}/>

          <Label title={"Alert Trigger Count"}>
              <Input
                disabled
                style={{ width: "45%", float: "left" }}
                value={this.state.trigger_count}
              />
            </Label>


          <Label title={"Trigger Delay (sec)"}>
            <Input
              value={this.state.trigger_delay_sec}
              id="trigger_delay_sec"
              onChange= {(event) => onUpdateSetStateValue.bind(this)(event,"trigger_delay_sec")}
              onKeyDown= {(event) => onEnterSendFloatValue.bind(this)(event,appNamespace + "/set_trigger_delay")}
              style={{ width: "80%" }}
            />
      </Label>

          <Label title="Enable Snapshot Triggers">
              <Toggle
              checked={this.state.snapshot_enabled===true}
              onClick={() => sendBoolMsg(appNamespace + "/enable_snapshot_trigger",!this.state.snapshot_enabled)}>
              </Toggle>
        </Label>

        <Label title="Enable Event Triggers">
              <Toggle
              checked={this.state.event_enabled===true}
              onClick={() => sendBoolMsg(appNamespace + "/enable_event_trigger",!this.state.event_enabled)}>
              </Toggle>
        </Label>




        </Column>
        </Columns>

        </div>

      <div style={{ borderTop: "1px solid #ffffff", marginTop: Styles.vars.spacing.medium, marginBottom: Styles.vars.spacing.xs }}/>

      <Columns>
          <Column>

            <ButtonMenu>
              <Button onClick={() => sendTriggerMsg( appNamespace + "/reset_app")}>{"Reset App"}</Button>
            </ButtonMenu>

            </Column>
          <Column>

              <ButtonMenu>
                <Button onClick={() => sendTriggerMsg(appNamespace + "/save_config")}>{"Save Config"}</Button>
          </ButtonMenu>

          </Column>
          <Column>

          <ButtonMenu>
                <Button onClick={() => sendTriggerMsg( appNamespace + "/reset_config")}>{"Reset Config"}</Button>
          </ButtonMenu>

  
          </Column>
        </Columns>



      </Column>
        </Columns>

      </Section>

    
    )
  }


  render() {
    if (this.state.needs_update === true){
      this.setState({needs_update: false})
    }
    const connected = this.state.connected === true
    const appNamespace = (connected) ? this.getAppNamespace() : null
    const imageNamespace = appNamespace + '/' + this.state.image_name

    return (

      <Columns>
      <Column equalWidth={true}>

      <div hidden={!connected}>

      <NepiIFSaveData
        saveNamespace={appNamespace}
        title={"Nepi_IF_SaveData"}
      />

      </div>

      <ImageViewer
        imageTopic={imageNamespace}
        title={this.state.image_name}
        hideQualitySelector={false}
      />


      </Column>
      <Column>



      {this.renderApp()}



      </Column>
      </Columns>

      )
    }  

}
export default AppAiAlerts
