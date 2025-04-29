:::{image} _static/logo.svg
:class: only-light
:alt: ESSsans
:width: 60%
:align: center
:::
:::{image} _static/logo-dark.svg
:class: only-dark
:alt: ESSsans
:width: 60%
:align: center
:::

```{raw} html
   <style>
    .transparent {display: none; visibility: hidden;}
    .transparent + a.headerlink {display: none; visibility: hidden;}
   </style>
```

```{role} transparent
```

# {transparent}`ESSsans`

<span style="font-size:1.2em;font-style:italic;color:var(--pst-color-text-muted);text-align:center;">
  SANS data reduction for the European Spallation Source
  </br></br>
</span>

## Quick links

::::{grid} 3

:::{grid-item-card} LoKI
:link: loki/index.md
:img-bottom: ../_static/previews/loki.png

:::

:::{grid-item-card} Skadi
:link: skadi/index.md

:::

:::{grid-item-card} ISIS instruments
:link: isis/index.md

:::

::::

::::{grid} 3

::::{grid-item-card} Common tools
:link: common/index.md

:::

::::

## Installation

To install ESSsans and all of its dependencies, use

`````{tab-set}
````{tab-item} pip
```sh
pip install esssans
```
````
````{tab-item} conda
```sh
conda install -c conda-forge -c scipp esssans
```
````
`````

## Get in touch

- If you have questions that are not answered by these documentation pages, ask on [discussions](https://github.com/scipp/esssans/discussions). Please include a self-contained reproducible example if possible.
- Report bugs (including unclear, missing, or wrong documentation!), suggest features or view the source code [on GitHub](https://github.com/scipp/esssans).

```{toctree}
---
hidden:
---

loki/index
skadi/index
isis/index
common/index
api-reference/index
developer/index
about/index
```
