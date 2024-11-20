# Latest Research

<style>
    .full-img .nt-card .nt-card-image img {
        height: 300px; /* or whatever height suits your design */
    }
    .full-img .nt-card .nt-card-image {
        min-height: 100%;
        /*aspect-ratio: 16 / 9;  adjust to your needs */
    }
</style>

::cards:: cols=1 class_name="full-img"

- title: "RPN 2:  On Interdependence Function Learning (November 2024)"
  content: | 
    This paper proposes a redesign of the RPN architecture, introducing the new RPN 2 (i.e., 
    Reconciled Polynomial Network version 2.0) model. As illustrate by Figure 1, RPN 2 incorporates a 
    novel component, the interdependence functions, to explicitly model diverse relationships among 
    both data instances and attributes. While we refer to this component as “interdependence”, this
    function actually captures a wide range of relationships within the input data, including structural
    interdependence, logical causality, statistical correlation, and numerical similarity or dissimilarity
  image:
    url: ../assets/img/rpn2.png
    height: 500
  url: "https://arxiv.org/abs/2411.11162"

- title: "RPN: Reconciled Polynomial Network (July 2024)"
  content: | 
    This paper proposes the task of "deep function learning" and introduce a novel deep function learning base model,
    i.e., the Reconciled Polynomial Network (RPN).<br>
    RPN has a versatile model architecture and attains superior modeling capabilities for diverse deep function 
    learning tasks on various multi-modality datasets.
    RPN also provides a canonical representation for many existing machine learning and deep learning models, 
    including but not limited to PGMs, kernel SVM, MLP and KAN.
  image:
    url: ../overrides/img/background.png
    height: 500
  url: "https://arxiv.org/abs/2407.04819"


::/cards::

<!--
## Recent Research

::cards:: cols=2

- title: Zeus
  content: Lorem ipsum dolor sit amet.
  image: ../overrides/img/background.png
  url: https://en.wikipedia.org/wiki/Zeus

- title: Athena
  content: Lorem ipsum dolor sit amet.
  image: ../assets/img/logo_white.png

- title: Poseidon
  content: Lorem ipsum dolor sit amet.
  image: ../assets/img/logo_white.png

::/cards::


## Past Research

::cards:: cols=4

- title: Zeus
  content: Lorem ipsum dolor sit amet.
  image: ../overrides/img/background.png
  url: https://en.wikipedia.org/wiki/Zeus

- title: Athena
  content: Lorem ipsum dolor sit amet.
  image: ../assets/img/logo_white.png

- title: Poseidon
  content: Lorem ipsum dolor sit amet.
  image: ../assets/img/logo_white.png

- title: Artemis
  content: Lorem ipsum dolor sit amet.
  image: ../assets/img/logo_white.png

- title: Ares
  content: Lorem ipsum dolor sit amet.
  image: ../assets/img/logo_white.png

- title: Nike
  content: Lorem ipsum dolor sit amet.
  image: ../assets/img/logo_white.png

::/cards::
-->

<!-- https://www.neoteroi.dev/mkdocs-plugins/cards/ -->