# waschmaschine

Predict the remaining washing time.

![grafana](grafana.png)

## Background

Power is measured e.g. by [Gosund SP1](https://templates.blakadder.com/gosund_SP1.html) or SP112. Tasmota firmware is flashed onto the ESP chip and data is provided via MQTT. I use [FHEM](https://fhem.de/) as home automation server.

## Features

* Reads watt values from MySQL db and writes into pkl file
* Determine washing sessions (cleaning and clustering the records)
* Train a model
* Store the model in file (pkl)

Open (not yet implemented):

* Predict the remaining washing time during runtime
