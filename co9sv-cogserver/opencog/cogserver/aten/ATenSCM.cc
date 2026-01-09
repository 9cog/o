/*
 * ATenSCM.cc
 *
 * Scheme bindings for ATen tensor operations
 * Copyright (c) 2025 OpenCog Foundation
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License v3 as
 * published by the Free Software Foundation and including the exceptions
 * at http://opencog.org/wiki/Licenses
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program; if not, write to:
 * Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#ifdef HAVE_ATEN
#ifdef HAVE_GUILE

#include <libguile.h>
#include <opencog/guile/SchemePrimitive.h>
#include <opencog/guile/SchemeSmob.h>

#include "TensorValue.h"
#include "ATenSpace.h"

using namespace opencog;

namespace opencog {

class ATenSCM
{
private:
	static void* init_in_guile(void*);
	static void init_in_module(void*);
	void init(void);

	// Tensor creation
	ValuePtr do_create_tensor(SCM, SCM, SCM, SCM);

	// Tensor operations
	ValuePtr do_matmul(ValuePtr, ValuePtr);
	ValuePtr do_elementwise(ValuePtr, ValuePtr, const std::string&);
	ValuePtr do_activate(ValuePtr, const std::string&);
	ValuePtr do_softmax(ValuePtr, int);
	ValuePtr do_reshape(ValuePtr, SCM);
	ValuePtr do_transpose(ValuePtr, int, int);
	SCM do_shape(ValuePtr);

	// Similarity
	double do_cosine_similarity(ValuePtr, ValuePtr);
	double do_dot(ValuePtr, ValuePtr);
	double do_norm(ValuePtr, int);
	ValuePtr do_normalize(ValuePtr);

	// Conversion
	ValuePtr do_to_floatvalue(ValuePtr);
	ValuePtr do_from_floatvalue(ValuePtr, SCM);
	SCM do_to_list(ValuePtr);

	// System info
	bool do_cuda_available(void);
	int do_cuda_device_count(void);

	// Helper to convert SCM list to vector<int64_t>
	std::vector<int64_t> scm_to_shape(SCM);

public:
	ATenSCM(void);
	~ATenSCM();
};

// ============================================================
// Constructor

ATenSCM::ATenSCM(void)
{
	static bool is_init = false;
	if (is_init) return;
	is_init = true;
	scm_with_guile(init_in_guile, this);
}

ATenSCM::~ATenSCM() {}

void* ATenSCM::init_in_guile(void* self)
{
	scm_c_define_module("opencog aten", init_in_module, self);
	scm_c_use_module("opencog aten");
	return nullptr;
}

void ATenSCM::init_in_module(void* data)
{
	ATenSCM* self = (ATenSCM*) data;
	self->init();
}

void ATenSCM::init(void)
{
	// Tensor creation
	define_scheme_primitive("aten-create-tensor",
		&ATenSCM::do_create_tensor, this, "aten");

	// Operations
	define_scheme_primitive("aten-matmul",
		&ATenSCM::do_matmul, this, "aten");
	define_scheme_primitive("aten-elementwise",
		&ATenSCM::do_elementwise, this, "aten");
	define_scheme_primitive("aten-activate",
		&ATenSCM::do_activate, this, "aten");
	define_scheme_primitive("aten-softmax",
		&ATenSCM::do_softmax, this, "aten");
	define_scheme_primitive("aten-reshape",
		&ATenSCM::do_reshape, this, "aten");
	define_scheme_primitive("aten-transpose",
		&ATenSCM::do_transpose, this, "aten");
	define_scheme_primitive("aten-shape",
		&ATenSCM::do_shape, this, "aten");

	// Similarity
	define_scheme_primitive("aten-cosine-similarity",
		&ATenSCM::do_cosine_similarity, this, "aten");
	define_scheme_primitive("aten-dot",
		&ATenSCM::do_dot, this, "aten");
	define_scheme_primitive("aten-norm",
		&ATenSCM::do_norm, this, "aten");
	define_scheme_primitive("aten-normalize",
		&ATenSCM::do_normalize, this, "aten");

	// Conversion
	define_scheme_primitive("aten-to-floatvalue",
		&ATenSCM::do_to_floatvalue, this, "aten");
	define_scheme_primitive("aten-from-floatvalue",
		&ATenSCM::do_from_floatvalue, this, "aten");
	define_scheme_primitive("aten-to-list",
		&ATenSCM::do_to_list, this, "aten");

	// System info
	define_scheme_primitive("aten-cuda-available",
		&ATenSCM::do_cuda_available, this, "aten");
	define_scheme_primitive("aten-cuda-device-count",
		&ATenSCM::do_cuda_device_count, this, "aten");
}

// ============================================================
// Helper

std::vector<int64_t> ATenSCM::scm_to_shape(SCM sshape)
{
	std::vector<int64_t> shape;
	while (scm_is_pair(sshape)) {
		SCM elem = scm_car(sshape);
		shape.push_back(scm_to_int64(elem));
		sshape = scm_cdr(sshape);
	}
	return shape;
}

// ============================================================
// Tensor creation

ValuePtr ATenSCM::do_create_tensor(SCM sshape, SCM sdtype, SCM sinit, SCM sdevice)
{
	std::vector<int64_t> shape = scm_to_shape(sshape);
	std::string dtype = scm_to_locale_string(sdtype);
	std::string init = scm_to_locale_string(sinit);

	at::ScalarType scalar_type = ATenSpace::parse_dtype(dtype);

	auto options = torch::TensorOptions().dtype(scalar_type);

	at::Tensor t;
	if (init == "zeros") {
		t = torch::zeros(shape, options);
	} else if (init == "ones") {
		t = torch::ones(shape, options);
	} else if (init == "randn") {
		t = torch::randn(shape, options);
	} else if (init == "xavier") {
		t = torch::empty(shape, options);
		if (shape.size() >= 2) torch::nn::init::xavier_uniform_(t);
	} else if (init == "kaiming") {
		t = torch::empty(shape, options);
		if (shape.size() >= 2) torch::nn::init::kaiming_uniform_(t);
	} else {
		throw RuntimeException(TRACE_INFO, "Unknown init: %s", init.c_str());
	}

	return createTensorValue(t);
}

// ============================================================
// Operations

ValuePtr ATenSCM::do_matmul(ValuePtr a, ValuePtr b)
{
	TensorValuePtr ta = TensorValueCast(a);
	TensorValuePtr tb = TensorValueCast(b);
	if (!ta || !tb)
		throw RuntimeException(TRACE_INFO, "Expected TensorValues");
	return createTensorValue(torch::matmul(ta->tensor(), tb->tensor()));
}

ValuePtr ATenSCM::do_elementwise(ValuePtr a, ValuePtr b, const std::string& op)
{
	TensorValuePtr ta = TensorValueCast(a);
	TensorValuePtr tb = TensorValueCast(b);
	if (!ta || !tb)
		throw RuntimeException(TRACE_INFO, "Expected TensorValues");

	at::Tensor result;
	if (op == "add") result = ta->tensor() + tb->tensor();
	else if (op == "sub") result = ta->tensor() - tb->tensor();
	else if (op == "mul") result = ta->tensor() * tb->tensor();
	else if (op == "div") result = ta->tensor() / tb->tensor();
	else throw RuntimeException(TRACE_INFO, "Unknown op: %s", op.c_str());

	return createTensorValue(result);
}

ValuePtr ATenSCM::do_activate(ValuePtr v, const std::string& activation)
{
	TensorValuePtr tv = TensorValueCast(v);
	if (!tv) throw RuntimeException(TRACE_INFO, "Expected TensorValue");

	at::Tensor result;
	if (activation == "relu") result = torch::relu(tv->tensor());
	else if (activation == "sigmoid") result = torch::sigmoid(tv->tensor());
	else if (activation == "tanh") result = torch::tanh(tv->tensor());
	else if (activation == "gelu") result = torch::gelu(tv->tensor());
	else throw RuntimeException(TRACE_INFO, "Unknown activation: %s", activation.c_str());

	return createTensorValue(result);
}

ValuePtr ATenSCM::do_softmax(ValuePtr v, int dim)
{
	TensorValuePtr tv = TensorValueCast(v);
	if (!tv) throw RuntimeException(TRACE_INFO, "Expected TensorValue");
	if (dim < 0) dim = tv->tensor().dim() - 1;
	return createTensorValue(torch::softmax(tv->tensor(), dim));
}

ValuePtr ATenSCM::do_reshape(ValuePtr v, SCM sshape)
{
	TensorValuePtr tv = TensorValueCast(v);
	if (!tv) throw RuntimeException(TRACE_INFO, "Expected TensorValue");
	std::vector<int64_t> shape = scm_to_shape(sshape);
	return createTensorValue(tv->tensor().reshape(shape));
}

ValuePtr ATenSCM::do_transpose(ValuePtr v, int dim0, int dim1)
{
	TensorValuePtr tv = TensorValueCast(v);
	if (!tv) throw RuntimeException(TRACE_INFO, "Expected TensorValue");
	return createTensorValue(tv->tensor().transpose(dim0, dim1));
}

SCM ATenSCM::do_shape(ValuePtr v)
{
	TensorValuePtr tv = TensorValueCast(v);
	if (!tv) throw RuntimeException(TRACE_INFO, "Expected TensorValue");

	SCM result = SCM_EOL;
	auto shape = tv->shape();
	for (auto it = shape.rbegin(); it != shape.rend(); ++it) {
		result = scm_cons(scm_from_int64(*it), result);
	}
	return result;
}

// ============================================================
// Similarity

double ATenSCM::do_cosine_similarity(ValuePtr a, ValuePtr b)
{
	TensorValuePtr ta = TensorValueCast(a);
	TensorValuePtr tb = TensorValueCast(b);
	if (!ta || !tb)
		throw RuntimeException(TRACE_INFO, "Expected TensorValues");

	at::Tensor a_flat = ta->tensor().flatten().to(at::kDouble);
	at::Tensor b_flat = tb->tensor().flatten().to(at::kDouble);
	at::Tensor dot = torch::dot(a_flat, b_flat);
	at::Tensor norm_a = torch::norm(a_flat);
	at::Tensor norm_b = torch::norm(b_flat);
	return (dot / (norm_a * norm_b + 1e-8)).item<double>();
}

double ATenSCM::do_dot(ValuePtr a, ValuePtr b)
{
	TensorValuePtr ta = TensorValueCast(a);
	TensorValuePtr tb = TensorValueCast(b);
	if (!ta || !tb)
		throw RuntimeException(TRACE_INFO, "Expected TensorValues");

	at::Tensor a_flat = ta->tensor().flatten().to(at::kDouble);
	at::Tensor b_flat = tb->tensor().flatten().to(at::kDouble);
	return torch::dot(a_flat, b_flat).item<double>();
}

double ATenSCM::do_norm(ValuePtr v, int p)
{
	TensorValuePtr tv = TensorValueCast(v);
	if (!tv) throw RuntimeException(TRACE_INFO, "Expected TensorValue");
	return torch::norm(tv->tensor(), p).item<double>();
}

ValuePtr ATenSCM::do_normalize(ValuePtr v)
{
	TensorValuePtr tv = TensorValueCast(v);
	if (!tv) throw RuntimeException(TRACE_INFO, "Expected TensorValue");
	at::Tensor n = torch::norm(tv->tensor());
	return createTensorValue(tv->tensor() / (n + 1e-8));
}

// ============================================================
// Conversion

ValuePtr ATenSCM::do_to_floatvalue(ValuePtr v)
{
	TensorValuePtr tv = TensorValueCast(v);
	if (!tv) throw RuntimeException(TRACE_INFO, "Expected TensorValue");
	return tv->to_float_value();
}

ValuePtr ATenSCM::do_from_floatvalue(ValuePtr v, SCM sshape)
{
	FloatValuePtr fv = FloatValueCast(v);
	if (!fv) throw RuntimeException(TRACE_INFO, "Expected FloatValue");
	std::vector<int64_t> shape = scm_to_shape(sshape);
	return createTensorValue(TensorValue::from_float_value(fv, shape).tensor());
}

SCM ATenSCM::do_to_list(ValuePtr v)
{
	TensorValuePtr tv = TensorValueCast(v);
	if (!tv) throw RuntimeException(TRACE_INFO, "Expected TensorValue");

	std::vector<double> data = tv->to_vector();
	SCM result = SCM_EOL;
	for (auto it = data.rbegin(); it != data.rend(); ++it) {
		result = scm_cons(scm_from_double(*it), result);
	}
	return result;
}

// ============================================================
// System info

bool ATenSCM::do_cuda_available(void)
{
	return ATenSpace::cuda_available();
}

int ATenSCM::do_cuda_device_count(void)
{
	return ATenSpace::cuda_device_count();
}

} // namespace opencog

// ============================================================
// Module initialization

extern "C" {
void opencog_aten_init(void)
{
	static ATenSCM aten_scm;
}
}

#endif // HAVE_GUILE
#endif // HAVE_ATEN
