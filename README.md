## The OpenCog Project 👋
[OpenCog aims to create AGI](https://wiki.opencog.org/w/The_Open_Cognition_Project)
with a combination of exploration, engineering and basic science research.
Side quests have included robotics systems ([Hanson Robotics](https://www.hansonrobotics.com)),
financial systems (Aidiya),
genomics (MOZI and [Rejuve.bio](https://www.rejuve.bio)),
machine learning ([predicting risk from clinician notes](https://doi.org/10.1371/journal.pone.0085733)),
natural language chatbots ([virtual dog playing fetch](https://www.youtube.com/watch?v=FEmpGRLwbqE)) and more.
This project was pioneered by [Dr. Ben Goertzel](https://en.wikipedia.org/wiki/Ben_Goertzel).
Git repos fall into four categories:

### OpenCog AtomSpace
The core of the system. As of 2025, it is active, stable and supported.

* [AtomSpace](https://github.com/opencog/atomspace) - Hypergraph database and query engine.
* [Storage](https://github.com/opencog/atomspace-storage) - Base class for saving, loading, sending and receiving Atoms and AtomSpaces
* [CogServer](https://github.com/opencog/cogserver) and [atomspace-cog](https://github.com/opencog/atomspace-cog) - Networking, json, websockets.
* [atomspace-rocks](https://github.com/opencog/atomspace-rocks) - Disk I/O storage, based on RocksDB.
* [Proxy Nodes](https://wiki.opencog.org/w/ProxyNode) - Managing Atoms flowing through large Atomspaces. 
* [Sparse Vectors/Matrix](https://github.com/opencog/matrix) - Working with graphs as (embeddings in) sparse vectors.
* [Link Grammar](https://github.com/opencog/link-grammar) - Maximal Planar Graph (MPG) parsing, natural lanuage parsing (NLP).
* [Docker containers](https://github.com/opencog/docker) - System integration and demos.
* [atomspace-pgres](https://github.com/opencog/atomspace-pgres) - Postgres StorageNode. Works, but old, deprecated.

### OpenCog Research
Git repos in which active resarch is being carried out:
* [Sensory](https://github.com/opencog/sensory) - Dataflow of graphlets to/from external world. Agents I/O system.
* [Atomese-SIMD](https://github.com/opencog/atomese-simd) - Flowing data to GPU's and other SIMD (OpenCL/CUDA) hardware w/the sensory API.
* [Learn](https://github.com/opencog/learn) - Symbolic learning ("mature", batch-based processing.)
* [Agents](https://github.com/opencog/agents) - Refactoring learning for an interactive environment.
* [Motor](https://github.com/opencog/motor) - Controlling the focus of sensory attention. Perception-action.

### OpenCog Fossils
Older, abandoned and obsolete components and experiments. These were attempts to build subsystems 
with specific goals and ideas in mind. As experiments, they provided validation for certain design
ideas. They were educational and fun, but turned out to be unworkable. Thus, development has
halted. These projects are no longer maintained. They do contain useful subsystems that could be
salvaged for future use. This includes:
* PLN, URE, Attention, Ghost, Relex, R2L, ROS, Hanson Robotics Eva/Sophia
* MOSES (but not as-moses, see below).
* Any repo that is marked "read-only" or "obsolete".

### OpenCog Hyperon
Being developed by [Singularity.net](https://singularitynet.io).

### OpenCog Incubator
These are the immature, incomplete, promising projects that haven't taken off yet.

* [as-moses](https://github.com/opencog/as-moses) - Port of MOSES to the AtomSpace.
* [SQL Bridge](https://github.com/opencog/atomspace-bridge) - Direct I/O between SQL and AtomSpace
* [Prolog-on-Atomspace](https://github.com/opencog/atomspace/tree/master/opencog/persist/prolog) - proof-of-concept
* [Chemistry](https://github.com/opencog/cheminformatics) - Molecular bonds, molecular structural formulas (proof-of-concpept.)
* [agi-bio](https://github.com/opencog/agi-bio) - Genomics, proteomics system used by MOZI and rejuve.bio
* [Vision](https://github.com/opencog/vision) - Extracting structure from images, video (proof-of-concept.)
* [Hyperon-on-top-of-atomspace](https://github.com/opencog/atomspace-metta) - Hyperon backwards-compat layer (proof-of-concept.)
* [SpaceTime](https://github.com/opencog/spacetime) - Octree spatial bounding boxes and time intervals in Atomese.

# HELP WANTED
The above-mentioned commercial projects don't pay the bills. There are far more ideas
and possibilities than there is time or money. If you're a software developer, bored
and looking for something to do, there's a lot of great stuff here that is worthy of
attention. If you are an academic, scientist or grad student, someone who wants to do
cross-over Symbolic AI and Deep-Learning Neural Net research, and need a base toolset,
this is the place. We will work with you to make sure this stuff fits your needs and
does what you want it to do, the way you want it.
Contact [Linas Vepstas](linasvepstas@gmail.com).

### Commercial support
If you are a commercial business looking to use any of these components in your products,
we can provide full-time support, if that's what you want. We'll custom-taylor components,
systems, and API's to suit your needs. If you are an investor looking to build up a venture,
well yes, that could happen too. Talk to us. Contact [Linas Vepstas](linasvepstas@gmail.com).
