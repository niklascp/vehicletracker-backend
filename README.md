# Vehicle Tracker: An open-source multi-model framework for public transport arrival prediction

This repository contain a novel framework for *multi-model* arrival prediction in public transport systems. With *multi-model*, we emphasize the ability for the framework to simultaneously maintain and supply predictions for the arrival of public transport services using multiple, and possibly overlapping, machine learning models. This ability allows the framework to act as an ensemble of the base models and provides both robustness, flexibility, and enhanced prediction accuracy. The framework automatically collect and monitor base model performance in order to weigh between each of the base models applicable for a given *spatial* and *temporal* reference context. 

![](img/vehicletracker-experimental-wide.svg)

The above illustrates the framework architecture which is building on proven methods for achieving a scalable and sustainable system and can be configured to consume computational power only when needed thus minimizing operational cost and climate footprint for public transport authorities using the system.

## Backend Nodes
This repository contain the backend of the framework, see [https://github.com/niklascp/vehicletracker-frontend](https://github.com/niklascp/vehicletracker-frontend) for the corresponding frontend written with AngularJS.

To start a vehicle tracker node, simply run the following, where ``configuration.yaml`` specifies which components to initiate on this node.
```
vehicletracker -c configuration.yaml
```
