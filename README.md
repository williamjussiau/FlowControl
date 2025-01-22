<!-- Ce que le projet fait
Pourquoi le projet est utile
Prise en main du projet par les utilisateurs
Où les utilisateurs peuvent obtenir de l’aide sur votre projet
Qui maintient et contribue au projet -->

# flowcontrol toolbox
The flowcontrol toolbox is an open-source toolbox addressing the simulation and control of 2D incompressible flows.

## Introduction
The primary goal of the toolbox was the implementation of feedback control design, but it may be used for a variety of other topics such as model reduction or identification, actuator and sensor placement... 

The toolbox implements two benchmarks for flow control and is designed to allow for easy implementation of new cases. To a certain extent, the toolbox proposes

The core of the toolbox is in Python and relies on the backend [FEniCS](https://fenicsproject.org/). Our roadmap includes updating the project to the newer version of FEniCS, namely [FEniCSx](https://fenicsproject.org/documentation/).

## What the toolbox proposes

### Two benchmarks

#### Cylinder

#### Cavity
![Snapshot of the attractor of the flow over an open cavity at Re=7500](/illustration/cavity_attractor.png)


### Define new benchmark
Modify parameters: Re, actuators, sensors
Or implement a completely new case

No meshing tools are shipped: do it yourself, gmsh is suggested 

### Uses of the tool
* Compute operators
* Can be used as backend in an optimization tool (as in [my article on control](https://link_to_article))

### Be very free to extend the capability

## Roadmap
Release of control tools, optimization

## Getting help
Send message at william.jussiau@gmail.com

[FEniCS documentation](https://olddocs.fenicsproject.org/dolfin/2019.1.0/) and [FEniCS forum](https://fenicsproject.discourse.group/)

## About me
I'm Mona, an Octocat and GitHub mascot! 
![Mona the Octocat in the style of Rosie the Riveter. Mona is wearing blue coveralls and a red and white polka dot hairscarf, on a background of a yellow circle outlined in blue. She is holding a wrench in one tentacle, and flexing her muscles. Text says "We can do it!"]([https://octodex.github.com/images/welcometocat.png](https://octodex.github.com/images/mona-the-rivetertocat.png))

## Contact me
Find me over on the [GitHub Blog](https://github.blog/) or on the [GitHub Community Discussions](https://github.com/orgs/community/discussions)

This README has been optimized for accessibility based on GitHub's blogpost "[Tips for Making your GitHub Profile Page Accessible](https://github.blog/2023-10-26-5-tips-for-making-your-github-profile-page-accessible)".


***
---
## Emphasis

**This is bold text**

__This is bold text__

*This is italic text*

_This is italic text_


## Lists

Unordered

+ Create a list by starting a line with `+`, `-`, or `*`
+ Sub-lists are made by indenting 2 spaces:
  - Marker character change forces new list start:
    * Ac tristique libero volutpat at
    * Facilisis in pretium nisl aliquet
    * Nulla volutpat aliquam velit
+ Very easy!

Ordered

1. Lorem ipsum dolor sit amet
2. Consectetur adipiscing elit
3. Integer molestie lorem at massa


1. You can use sequential numbers...
1. ...or keep all the numbers as `1.`

Start numbering with offset:

57. foo
1. bar


## Code

Inline `code`

Indented code

    // Some comments
    line 1 of code
    line 2 of code
    line 3 of code


Block code "fences"

```
Sample text here...
```

Syntax highlighting

``` py
def foo(bar: int=1)
  return bar + 1

print(foo(4))
```

## Tables

| Option | Description |
| ------ | ----------- |
| data   | path to data files to supply the data that will be passed into templates. |
| engine | engine to be used for processing templates. Handlebars is the default. |
| ext    | extension to be used for dest files. |

Right aligned columns

| Option | Description |
| ------:| -----------:|
| data   | path to data files to supply the data that will be passed into templates. |
| engine | engine to be used for processing templates. Handlebars is the default. |
| ext    | extension to be used for dest files. |


## Links

[link text](http://dev.nodeca.com)

[link with title](http://nodeca.github.io/pica/demo/ "title text!")

Autoconverted link https://github.com/nodeca/pica (enable linkify to see)


## Images

![Minion](https://octodex.github.com/images/minion.png)
![Stormtroopocat](https://octodex.github.com/images/stormtroopocat.jpg "The Stormtroopocat")

Like links, Images also have a footnote style syntax

![Alt text][id]

With a reference later in the document defining the URL location:

[id]: https://octodex.github.com/images/dojocat.jpg  "The Dojocat"


## Plugins

The killer feature of `markdown-it` is very effective support of
[syntax plugins](https://www.npmjs.org/browse/keyword/markdown-it-plugin).


### [Emojies](https://github.com/markdown-it/markdown-it-emoji)

> Classic markup: :wink: :cry: :laughing: :yum:
>
> Shortcuts (emoticons): :-) :-( 8-) ;)

see [how to change output](https://github.com/markdown-it/markdown-it-emoji#change-output) with twemoji.


### [Subscript](https://github.com/markdown-it/markdown-it-sub) / [Superscript](https://github.com/markdown-it/markdown-it-sup)

- 19^th^
- H~2~O


### [\<ins>](https://github.com/markdown-it/markdown-it-ins)

++Inserted text++


### [\<mark>](https://github.com/markdown-it/markdown-it-mark)

==Marked text==


### [Footnotes](https://github.com/markdown-it/markdown-it-footnote)

Footnote 1 link[^first].

Footnote 2 link[^second].

Inline footnote^[Text of inline footnote] definition.

Duplicated footnote reference[^second].

[^first]: Footnote **can have markup**

    and multiple paragraphs.

[^second]: Footnote text.


### [Definition lists](https://github.com/markdown-it/markdown-it-deflist)

Term 1

:   Definition 1
with lazy continuation.

Term 2 with *inline markup*

:   Definition 2

        { some code, part of Definition 2 }

    Third paragraph of definition 2.

_Compact style:_

Term 1
  ~ Definition 1

Term 2
  ~ Definition 2a
  ~ Definition 2b


### [Abbreviations](https://github.com/markdown-it/markdown-it-abbr)

This is HTML abbreviation example.

It converts "HTML", but keep intact partial entries like "xxxHTMLyyy" and so on.

*[HTML]: Hyper Text Markup Language

### [Custom containers](https://github.com/markdown-it/markdown-it-container)

::: warning
*here be dragons*
:::
