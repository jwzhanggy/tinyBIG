# Library Updates

- 2024-05-03: `tinybig` project launched.
- 2024-07-01: `tinybig` v0.1.0 released.
- 2024-07-09: `tinybig` v0.1.1 released.
    - Reflecting changes to make all the examples code
    - Changing the utility functions for handle diverse pre-, post-, output-, and activation processing functions.
    - Adding the device handling code to the model configs
- 2024-11-18: `tinybig` v0.2.0 released.
    - Redesigning the RPN framework for interdependent data modeling.
    - Adding the data interdependence functions.
    - Adding the data compression functions and fusion functions.
    - Updating the existing list of expansion, reconciliation and remainder functions.

-----------------------

::gantt::
[
    {
        "title": "Milestones",
        "events": [
            {
                "title": "tinybig Project Launched",
                "time": "2024-05-03",
                "icon": ":octicons-rocket-16:"
            },
            {
                "title": "tinybig v0.1.0 Released",
                "time": "2024-07-01",
                "icon": ":octicons-sun-16:"
            },
            {
                "title": "tinybig v0.2.0 Released",
                "time": "2024-11-18",
                "icon": ":octicons-sun-16:"
            }
        ]
    },
    {
        "title": "tinybig library project",
        "activities": [
            {
                "title": "brain-storming",
                "start": "2024-05-3",
                "lasts": "1 week"
            },
            {
                "title": "preliminary demo",
                "start": "2024-05-10",
                "lasts": "2 weeks"
            },
            {
                "title": "RPN evaluation with tinybig",
                "start": "2024-6-1",
                "lasts": "2 weeks"
            },
            {
                "title": "tinybig library polishing",
                "start": "2024-6-15",
                "lasts": "2 weeks"
            },
            {
                "title": "RPN 2 model designing",
                "start": "2024-8-01",
                "lasts": "4 weeks"
            },
            {
                "title": "tinybig 0.2.0 implementation",
                "start": "2024-9-01",
                "lasts": "4 weeks"
            },
            {
                "title": "model and toolkit polishing",
                "start": "2024-10-01",
                "lasts": "2 weeks"
            },
            {
                "title": "technical report writing",
                "start": "2024-10-15",
                "lasts": "5 weeks"
            },
        ]
    }
]
::/gantt::