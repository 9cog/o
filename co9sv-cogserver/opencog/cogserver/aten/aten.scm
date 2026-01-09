;
; OpenCog ATen/PyTorch tensor integration
;
; This module provides Scheme bindings for ATen tensor operations,
; enabling neuro-symbolic AI by bridging AtomSpace with PyTorch tensors.
;
; Copyright (c) 2025 OpenCog Foundation
;

(define-module (opencog aten))

(use-modules (opencog))
(use-modules (opencog exec))
(use-modules (ice-9 optargs))
(use-modules (srfi srfi-1))

; Load the ATen extension library if available
; Note: This will be conditionally available based on HAVE_ATEN
(define aten-available #f)

(catch #t
  (lambda ()
    (load-extension
      (string-append opencog-ext-path-aten "libaten")
      "opencog_aten_init")
    (set! aten-available #t))
  (lambda (key . args)
    (display "ATen extension not available. Install PyTorch and rebuild with -DWITH_ATEN=ON\n")))

(export aten-available)

;; ============================================================
;; Tensor Creation
;; ============================================================

(define* (cog-new-tensor shape #:key
                         (dtype 'float32)
                         (init 'zeros)
                         (device 'cpu))
"
  cog-new-tensor SHAPE
  cog-new-tensor SHAPE #:dtype DTYPE #:init INIT #:device DEVICE

  Create a new tensor with the given shape.

  SHAPE is a list of integers specifying dimensions, e.g., '(3 4) for 3x4 matrix.

  Optional keyword arguments:
    #:dtype  - Data type: 'float32 (default), 'float64, 'float16, 'int32, 'int64
    #:init   - Initialization: 'zeros (default), 'ones, 'randn, 'xavier, 'kaiming
    #:device - Device: 'cpu (default), 'cuda, or integer for specific GPU

  Example:
    (cog-new-tensor '(3 4))                          ; 3x4 zero tensor
    (cog-new-tensor '(128) #:init 'randn)            ; Random 128-dim vector
    (cog-new-tensor '(64 64) #:dtype 'float16 #:device 'cuda)

  Returns a TensorValue that can be attached to atoms.
"
  (if (not aten-available)
    (throw 'aten-error "ATen not available"))
  ; Implementation provided by C++ extension
  (aten-create-tensor shape
                      (symbol->string dtype)
                      (symbol->string init)
                      (if (number? device) device (symbol->string device))))

(export cog-new-tensor)

;; ============================================================
;; Tensor Storage on Atoms
;; ============================================================

(define (cog-set-tensor! atom key tensor)
"
  cog-set-tensor! ATOM KEY TENSOR

  Store a tensor on an atom at the given key.

  ATOM   - The atom to attach the tensor to
  KEY    - The key (usually a PredicateNode) for this tensor
  TENSOR - A TensorValue or list of numbers (will be converted)

  Example:
    (define emb (cog-new-tensor '(128) #:init 'randn))
    (cog-set-tensor! (Concept \"cat\") (Predicate \"embedding\") emb)
"
  (if (not aten-available)
    (throw 'aten-error "ATen not available"))
  (cog-set-value! atom key tensor))

(export cog-set-tensor!)

(define (cog-get-tensor atom key)
"
  cog-get-tensor ATOM KEY

  Retrieve a tensor stored on an atom.

  Returns the TensorValue if found, or #f if not present.

  Example:
    (cog-get-tensor (Concept \"cat\") (Predicate \"embedding\"))
"
  (if (not aten-available)
    (throw 'aten-error "ATen not available"))
  (cog-value atom key))

(export cog-get-tensor)

;; ============================================================
;; Tensor Operations
;; ============================================================

(define (cog-tensor-matmul a b)
"
  cog-tensor-matmul A B

  Matrix multiply two tensors.

  A and B are TensorValues. Returns a new TensorValue with the result.

  Example:
    (define w (cog-new-tensor '(128 64)))
    (define x (cog-new-tensor '(64)))
    (cog-tensor-matmul w x)  ; Returns 128-dim vector
"
  (if (not aten-available)
    (throw 'aten-error "ATen not available"))
  (aten-matmul a b))

(export cog-tensor-matmul)

(define (cog-tensor-add a b)
"
  cog-tensor-add A B

  Element-wise addition of two tensors.
"
  (if (not aten-available)
    (throw 'aten-error "ATen not available"))
  (aten-elementwise a b "add"))

(export cog-tensor-add)

(define (cog-tensor-sub a b)
"
  cog-tensor-sub A B

  Element-wise subtraction of two tensors.
"
  (if (not aten-available)
    (throw 'aten-error "ATen not available"))
  (aten-elementwise a b "sub"))

(export cog-tensor-sub)

(define (cog-tensor-mul a b)
"
  cog-tensor-mul A B

  Element-wise multiplication of two tensors.
"
  (if (not aten-available)
    (throw 'aten-error "ATen not available"))
  (aten-elementwise a b "mul"))

(export cog-tensor-mul)

(define (cog-tensor-div a b)
"
  cog-tensor-div A B

  Element-wise division of two tensors.
"
  (if (not aten-available)
    (throw 'aten-error "ATen not available"))
  (aten-elementwise a b "div"))

(export cog-tensor-div)

;; ============================================================
;; Activation Functions
;; ============================================================

(define (cog-tensor-relu tensor)
"
  cog-tensor-relu TENSOR

  Apply ReLU activation function.
"
  (if (not aten-available)
    (throw 'aten-error "ATen not available"))
  (aten-activate tensor "relu"))

(export cog-tensor-relu)

(define (cog-tensor-sigmoid tensor)
"
  cog-tensor-sigmoid TENSOR

  Apply sigmoid activation function.
"
  (if (not aten-available)
    (throw 'aten-error "ATen not available"))
  (aten-activate tensor "sigmoid"))

(export cog-tensor-sigmoid)

(define (cog-tensor-tanh tensor)
"
  cog-tensor-tanh TENSOR

  Apply tanh activation function.
"
  (if (not aten-available)
    (throw 'aten-error "ATen not available"))
  (aten-activate tensor "tanh"))

(export cog-tensor-tanh)

(define* (cog-tensor-softmax tensor #:optional (dim -1))
"
  cog-tensor-softmax TENSOR
  cog-tensor-softmax TENSOR DIM

  Apply softmax activation along dimension DIM (default: last dimension).
"
  (if (not aten-available)
    (throw 'aten-error "ATen not available"))
  (aten-softmax tensor dim))

(export cog-tensor-softmax)

;; ============================================================
;; Tensor Shape Operations
;; ============================================================

(define (cog-tensor-reshape tensor new-shape)
"
  cog-tensor-reshape TENSOR NEW-SHAPE

  Reshape tensor to new dimensions.

  Example:
    (cog-tensor-reshape (cog-new-tensor '(6)) '(2 3))
"
  (if (not aten-available)
    (throw 'aten-error "ATen not available"))
  (aten-reshape tensor new-shape))

(export cog-tensor-reshape)

(define* (cog-tensor-transpose tensor #:optional (dim0 0) (dim1 1))
"
  cog-tensor-transpose TENSOR
  cog-tensor-transpose TENSOR DIM0 DIM1

  Transpose tensor, swapping dimensions DIM0 and DIM1.
"
  (if (not aten-available)
    (throw 'aten-error "ATen not available"))
  (aten-transpose tensor dim0 dim1))

(export cog-tensor-transpose)

(define (cog-tensor-shape tensor)
"
  cog-tensor-shape TENSOR

  Get the shape of a tensor as a list.

  Example:
    (cog-tensor-shape (cog-new-tensor '(3 4)))  ; Returns (3 4)
"
  (if (not aten-available)
    (throw 'aten-error "ATen not available"))
  (aten-shape tensor))

(export cog-tensor-shape)

;; ============================================================
;; Similarity Operations
;; ============================================================

(define (cog-tensor-cosine-similarity a b)
"
  cog-tensor-cosine-similarity A B

  Compute cosine similarity between two tensors.
  Returns a number between -1 and 1.
"
  (if (not aten-available)
    (throw 'aten-error "ATen not available"))
  (aten-cosine-similarity a b))

(export cog-tensor-cosine-similarity)

(define (cog-tensor-dot a b)
"
  cog-tensor-dot A B

  Compute dot product of two tensors.
"
  (if (not aten-available)
    (throw 'aten-error "ATen not available"))
  (aten-dot a b))

(export cog-tensor-dot)

(define* (cog-tensor-norm tensor #:optional (p 2))
"
  cog-tensor-norm TENSOR
  cog-tensor-norm TENSOR P

  Compute Lp norm of tensor (default: L2 norm).
"
  (if (not aten-available)
    (throw 'aten-error "ATen not available"))
  (aten-norm tensor p))

(export cog-tensor-norm)

(define (cog-tensor-normalize tensor)
"
  cog-tensor-normalize TENSOR

  L2 normalize the tensor (divide by L2 norm).
"
  (if (not aten-available)
    (throw 'aten-error "ATen not available"))
  (aten-normalize tensor))

(export cog-tensor-normalize)

;; ============================================================
;; Conversion Utilities
;; ============================================================

(define (cog-tensor->floatvalue tensor)
"
  cog-tensor->floatvalue TENSOR

  Convert TensorValue to FloatValue (flattens to 1D).
  Useful for AtomSpace compatibility.
"
  (if (not aten-available)
    (throw 'aten-error "ATen not available"))
  (aten-to-floatvalue tensor))

(export cog-tensor->floatvalue)

(define (cog-floatvalue->tensor fv shape)
"
  cog-floatvalue->tensor FLOATVALUE SHAPE

  Convert FloatValue to TensorValue with specified shape.

  Example:
    (cog-floatvalue->tensor (FloatValue 1 2 3 4 5 6) '(2 3))
"
  (if (not aten-available)
    (throw 'aten-error "ATen not available"))
  (aten-from-floatvalue fv shape))

(export cog-floatvalue->tensor)

(define (cog-tensor->list tensor)
"
  cog-tensor->list TENSOR

  Convert tensor data to a flat Scheme list.
"
  (if (not aten-available)
    (throw 'aten-error "ATen not available"))
  (aten-to-list tensor))

(export cog-tensor->list)

;; ============================================================
;; System Information
;; ============================================================

(define (cog-aten-cuda-available?)
"
  cog-aten-cuda-available?

  Check if CUDA is available for GPU acceleration.
"
  (if (not aten-available)
    #f
    (aten-cuda-available)))

(export cog-aten-cuda-available?)

(define (cog-aten-cuda-device-count)
"
  cog-aten-cuda-device-count

  Get number of available CUDA devices.
"
  (if (not aten-available)
    0
    (aten-cuda-device-count)))

(export cog-aten-cuda-device-count)

;; ============================================================
;; High-level utilities
;; ============================================================

(define* (cog-embedding-similarity atom1 atom2 key)
"
  cog-embedding-similarity ATOM1 ATOM2 KEY

  Compute cosine similarity between embeddings stored on two atoms.

  Example:
    (cog-embedding-similarity
      (Concept \"cat\")
      (Concept \"dog\")
      (Predicate \"embedding\"))
"
  (let ((t1 (cog-get-tensor atom1 key))
        (t2 (cog-get-tensor atom2 key)))
    (if (and t1 t2)
      (cog-tensor-cosine-similarity t1 t2)
      (throw 'aten-error "One or both atoms missing tensor at key"))))

(export cog-embedding-similarity)

(define* (cog-nearest-by-embedding query-atom key candidates #:optional (k 10))
"
  cog-nearest-by-embedding QUERY-ATOM KEY CANDIDATES
  cog-nearest-by-embedding QUERY-ATOM KEY CANDIDATES K

  Find K nearest neighbors to QUERY-ATOM from CANDIDATES based on
  embedding similarity at KEY.

  Returns a list of (atom . similarity) pairs sorted by similarity.
"
  (if (not aten-available)
    (throw 'aten-error "ATen not available"))
  (let ((query-tensor (cog-get-tensor query-atom key)))
    (if (not query-tensor)
      (throw 'aten-error "Query atom has no tensor at key"))
    (let* ((scored (filter-map
                     (lambda (c)
                       (let ((t (cog-get-tensor c key)))
                         (if t
                           (cons c (cog-tensor-cosine-similarity query-tensor t))
                           #f)))
                     candidates))
           (sorted (sort scored (lambda (a b) (> (cdr a) (cdr b))))))
      (take sorted (min k (length sorted))))))

(export cog-nearest-by-embedding)

;; ============================================================
;; TensorLogic - Multi-Entity & Multi-Scale Network-Aware Operations
;; ============================================================
;;
;; The following functions provide high-level tensor logic operations
;; integrating ATen and ATenSpace patterns for neuro-symbolic AI.
;;

;; ---- Multi-Scale Operations ----

(define* (cog-multiscale-embed atom key #:key (store #t))
"
  cog-multiscale-embed ATOM KEY
  cog-multiscale-embed ATOM KEY #:store BOOL

  Compute multi-scale embeddings for ATOM at different granularities:
  - LOCAL: Direct node features
  - NEIGHBOR: 1-hop neighborhood aggregation
  - COMMUNITY: 2-hop cluster context
  - GLOBAL: Whole-graph summary

  If STORE is #t (default), stores result on the atom.

  Example:
    (cog-multiscale-embed (Concept \"cat\") (Predicate \"embedding\"))
"
  (if (not aten-available)
    (throw 'aten-error "ATen not available"))
  (aten-multiscale-compute atom key store))

(export cog-multiscale-embed)

(define* (cog-query-at-scale query-tensor scale #:optional (k 10))
"
  cog-query-at-scale QUERY-TENSOR SCALE
  cog-query-at-scale QUERY-TENSOR SCALE K

  Query atoms similar to QUERY-TENSOR at the specified SCALE level.
  SCALE is one of: 'local, 'neighbor, 'community, 'global

  Returns list of (atom . similarity) pairs.
"
  (if (not aten-available)
    (throw 'aten-error "ATen not available"))
  (aten-query-at-scale query-tensor (symbol->string scale) k))

(export cog-query-at-scale)

;; ---- Network-Aware Operations ----

(define* (cog-network-embed atom key #:optional (layers 2))
"
  cog-network-embed ATOM KEY
  cog-network-embed ATOM KEY LAYERS

  Compute network-aware embedding that incorporates graph structure.
  Uses LAYERS of graph neural network message passing.

  The embedding encodes:
  - Node features
  - Degree centrality
  - Type information
  - Aggregated neighbor information
"
  (if (not aten-available)
    (throw 'aten-error "ATen not available"))
  (aten-network-embed atom key layers))

(export cog-network-embed)

(define* (cog-propagate-embeddings atoms key #:optional (iterations 3))
"
  cog-propagate-embeddings ATOMS KEY
  cog-propagate-embeddings ATOMS KEY ITERATIONS

  Propagate embeddings through the graph via message passing.
  Updates embeddings for all ATOMS in place.
"
  (if (not aten-available)
    (throw 'aten-error "ATen not available"))
  (aten-propagate atoms key iterations))

(export cog-propagate-embeddings)

;; ---- Attention Operations (ECAN-style) ----

(define* (cog-stimulate atom #:optional (stimulus 1.0))
"
  cog-stimulate ATOM
  cog-stimulate ATOM STIMULUS

  Stimulate attention on ATOM, increasing its Short-Term Importance (STI).
  High-STI atoms are in the 'attentional focus' and get priority processing.
"
  (if (not aten-available)
    (throw 'aten-error "ATen not available"))
  (aten-stimulate atom stimulus))

(export cog-stimulate)

(define (cog-get-focus)
"
  cog-get-focus

  Get the current attentional focus - atoms with highest attention.
  Returns list of atoms sorted by STI (Short-Term Importance).
"
  (if (not aten-available)
    (throw 'aten-error "ATen not available"))
  (aten-get-focus))

(export cog-get-focus)

(define (cog-spread-attention source)
"
  cog-spread-attention SOURCE

  Spread attention from SOURCE atom to connected atoms.
  Uses both graph structure and Hebbian links for spreading.
"
  (if (not aten-available)
    (throw 'aten-error "ATen not available"))
  (aten-spread-attention source))

(export cog-spread-attention)

(define (cog-attention-cycle)
"
  cog-attention-cycle

  Run one attention allocation cycle:
  1. Decay STI (forgetting)
  2. Spread attention from focus atoms
  3. Update attentional focus
"
  (if (not aten-available)
    (throw 'aten-error "ATen not available"))
  (aten-attention-cycle))

(export cog-attention-cycle)

;; ---- Tensor Logic Inference ----

(define (cog-tensor-analogy A B C candidates)
"
  cog-tensor-analogy A B C CANDIDATES

  Solve analogy: A is to B as C is to ?

  Uses embedding arithmetic: D = C + (B - A)
  Returns the candidate most similar to the computed D.

  Example:
    ; king - man + woman = queen
    (cog-tensor-analogy
      (Concept \"man\") (Concept \"king\")
      (Concept \"woman\")
      (list (Concept \"queen\") (Concept \"princess\") (Concept \"duchess\")))
"
  (if (not aten-available)
    (throw 'aten-error "ATen not available"))
  (aten-analogy A B C candidates))

(export cog-tensor-analogy)

(define* (cog-tensor-unify pattern candidates #:optional (threshold 0.5))
"
  cog-tensor-unify PATTERN CANDIDATES
  cog-tensor-unify PATTERN CANDIDATES THRESHOLD

  Tensor-guided pattern unification.
  Finds CANDIDATES that match PATTERN based on embedding similarity.

  Returns list of binding maps for matches above THRESHOLD.
"
  (if (not aten-available)
    (throw 'aten-error "ATen not available"))
  (aten-unify pattern candidates threshold))

(export cog-tensor-unify)

;; ---- Multi-Entity Operations ----

(define* (cog-aggregate-entities entities key #:optional (method 'attention))
"
  cog-aggregate-entities ENTITIES KEY
  cog-aggregate-entities ENTITIES KEY METHOD

  Aggregate embeddings from multiple entities into a single representation.

  METHOD is one of:
  - 'mean     : Simple average
  - 'max      : Element-wise maximum
  - 'sum      : Element-wise sum
  - 'attention: Self-attention weighted (default)
  - 'deepsets : Permutation-invariant (sum + max concatenation)
"
  (if (not aten-available)
    (throw 'aten-error "ATen not available"))
  (aten-aggregate-entities entities key (symbol->string method)))

(export cog-aggregate-entities)

(define* (cog-cluster-entities entities key #:optional (k 5))
"
  cog-cluster-entities ENTITIES KEY
  cog-cluster-entities ENTITIES KEY K

  Cluster ENTITIES into K groups based on embedding similarity.
  Returns an association list mapping cluster IDs to entity lists.
"
  (if (not aten-available)
    (throw 'aten-error "ATen not available"))
  (aten-cluster-entities entities key k))

(export cog-cluster-entities)

; End of module
