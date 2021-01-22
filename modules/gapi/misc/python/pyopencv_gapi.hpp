#ifndef OPENCV_GAPI_PYOPENCV_GAPI_HPP
#define OPENCV_GAPI_PYOPENCV_GAPI_HPP

#ifdef HAVE_OPENCV_GAPI

#include <opencv2/gapi/cpu/gcpukernel.hpp>
#include <opencv2/gapi/python/python.hpp>

// NB: Python wrapper replaces :: with _ for classes
using gapi_GKernelPackage = cv::gapi::GKernelPackage;
using gapi_GNetPackage = cv::gapi::GNetPackage;
using gapi_ie_PyParams = cv::gapi::ie::PyParams;
using gapi_wip_IStreamSource_Ptr = cv::Ptr<cv::gapi::wip::IStreamSource>;

// NB: Python wrapper generate T_U for T<U>
// This behavior is only observed for inputs
using GOpaque_Size = cv::GOpaque<cv::Size>;

// FIXME: Python wrapper generate code without namespace std,
// so it cause error: "string wasn't declared"
// WA: Create using
using std::string;

template<>
bool pyopencv_to(PyObject* obj, std::vector<GCompileArg>& value, const ArgInfo& info)
{
    return pyopencv_to_generic_vec(obj, value, info);
}

template<>
PyObject* pyopencv_from(const std::vector<GCompileArg>& value)
{
    return pyopencv_from_generic_vec(value);
}

template<>
bool pyopencv_to(PyObject* obj, GRunArgs& value, const ArgInfo& info)
{
    return pyopencv_to_generic_vec(obj, value, info);
}

static PyObject* from_grunarg(const GRunArg& v)
{
    switch (v.index())
    {
        case GRunArg::index_of<cv::Mat>():
        {
            const auto& m = util::get<cv::Mat>(v);
            return pyopencv_from(m);
        }

        case GRunArg::index_of<cv::Scalar>():
        {
            const auto& s = util::get<cv::Scalar>(v);
            return pyopencv_from(s);
        }
        case GRunArg::index_of<cv::detail::VectorRef>():
        {
            const auto& vref = util::get<cv::detail::VectorRef>(v);
            switch (vref.getKind())
            {
                case cv::detail::OpaqueKind::CV_POINT2F:
                    return pyopencv_from(vref.rref<cv::Point2f>());
                case cv::detail::OpaqueKind::CV_RECT:
                    return pyopencv_from(vref.rref<cv::Rect>());
                default:
                    PyErr_SetString(PyExc_TypeError, "Unsupported kind for GArray");
                    return NULL;
            }
        }
        case GRunArg::index_of<cv::detail::OpaqueRef>():
        {
            const auto& oref = util::get<cv::detail::OpaqueRef>(v);
            switch (oref.getKind())
            {
                case cv::detail::OpaqueKind::CV_RECT:
                    return pyopencv_from(oref.rref<cv::Rect>());
                default:
                    PyErr_SetString(PyExc_TypeError, "Unsupported kind for GOpaque");
                    return NULL;
            }
        }
        default:
            PyErr_SetString(PyExc_TypeError, "Failed to unpack GRunArgs");
            return NULL;
    }
    GAPI_Assert(false);
}

template<>
PyObject* pyopencv_from(const GRunArgs& value)
{
    size_t i, n = value.size();

    // NB: It doesn't make sense to return list with a single element
    if (n == 1)
    {
        PyObject* item = from_grunarg(value[0]);
        if(!item)
        {
            return NULL;
        }
        return item;
    }

    PyObject* list = PyList_New(n);
    for(i = 0; i < n; ++i)
    {
        PyObject* item = from_grunarg(value[i]);
        if(!item)
        {
            Py_DECREF(list);
            PyErr_SetString(PyExc_TypeError, "Failed to unpack GRunArgs");
            return NULL;
        }
        PyList_SetItem(list, i, item);
    }

    return list;
}

template<>
bool pyopencv_to(PyObject* obj, GMetaArgs& value, const ArgInfo& info)
{
    return pyopencv_to_generic_vec(obj, value, info);
}

template<>
PyObject* pyopencv_from(const GMetaArgs& value)
{
    return pyopencv_from_generic_vec(value);
}

template <typename T>
static PyObject* extract_proto_args(PyObject* py_args, PyObject* kw)
{
    using namespace cv;

    GProtoArgs args;
    Py_ssize_t size = PyTuple_Size(py_args);
    args.reserve(size);
    for (int i = 0; i < size; ++i)
    {
        PyObject* item = PyTuple_GetItem(py_args, i);
        if (PyObject_TypeCheck(item, reinterpret_cast<PyTypeObject*>(pyopencv_GScalar_TypePtr)))
        {
            args.emplace_back(reinterpret_cast<pyopencv_GScalar_t*>(item)->v);
        }
        else if (PyObject_TypeCheck(item, reinterpret_cast<PyTypeObject*>(pyopencv_GMat_TypePtr)))
        {
            args.emplace_back(reinterpret_cast<pyopencv_GMat_t*>(item)->v);
        }
        else if (PyObject_TypeCheck(item, reinterpret_cast<PyTypeObject*>(pyopencv_GArrayP2f_TypePtr)))
        {
            args.emplace_back(reinterpret_cast<pyopencv_GArrayP2f_t*>(item)->v.strip());
        }
        else if (PyObject_TypeCheck(item, reinterpret_cast<PyTypeObject*>(pyopencv_GSize_TypePtr)))
        {
            args.emplace_back(reinterpret_cast<pyopencv_GSize_t*>(item)->v.strip());
        }
        else if (PyObject_TypeCheck(item, reinterpret_cast<PyTypeObject*>(pyopencv_GRects_TypePtr)))
        {
            args.emplace_back(reinterpret_cast<pyopencv_GRects_t*>(item)->v.strip());
        }
        else if (PyObject_TypeCheck(item, reinterpret_cast<PyTypeObject*>(pyopencv_GRect_TypePtr)))
        {
            args.emplace_back(reinterpret_cast<pyopencv_GRect_t*>(item)->v.strip());
        }
        else
        {
            PyErr_SetString(PyExc_TypeError, "Unsupported type for cv.GIn()/cv.GOut()");
            return NULL;
        }
    }

    return pyopencv_from<T>(T{std::move(args)});
}

static PyObject* pyopencv_cv_GIn(PyObject* , PyObject* py_args, PyObject* kw)
{
    return extract_proto_args<GProtoInputArgs>(py_args, kw);
}

static PyObject* pyopencv_cv_GOut(PyObject* , PyObject* py_args, PyObject* kw)
{
    return extract_proto_args<GProtoOutputArgs>(py_args, kw);
}

static PyObject* pyopencv_cv_gin(PyObject*, PyObject* py_args, PyObject* kw)
{
    using namespace cv;

    Py_INCREF(py_args);
    cv::ExtractArgsCallback callback([=](const cv::GTypesInfo& info) {
            // NB: This code will be executed from opencv_gapi.so
            PyGILState_STATE gstate;
            gstate = PyGILState_Ensure();

            cv::GRunArgs args;
            Py_ssize_t tuple_size = PyTuple_Size(py_args);
            args.reserve(tuple_size);

            for (int i = 0; i < tuple_size; ++i)
            {
                PyObject* item = PyTuple_GetItem(py_args, i);
                switch (info[i].shape)
                {
                    case cv::GShape::GMAT:
                    {
                        // NB: In case streaming it can be IStreamSource or cv::Mat
                        if (PyObject_TypeCheck(item,
                                    reinterpret_cast<PyTypeObject*>(pyopencv_gapi_wip_IStreamSource_TypePtr)))
                        {
                            cv::gapi::wip::IStreamSource::Ptr source =
                                reinterpret_cast<pyopencv_gapi_wip_IStreamSource_t*>(item)->v;
                            args.emplace_back(source);
                        }
                        else
                        {
                            cv::Mat m;
                            if (pyopencv_to(item, m, ArgInfo("mat", false)))
                            {
                                args.emplace_back(m);
                            }
                            else
                            {
                                util::throw_error(std::logic_error("Failed to obtain cv::Mat"));
                            }
                        }
                        break;
                    }
                    case cv::GShape::GARRAY:
                        util::throw_error(std::logic_error("GArray isn't supported for input"));
                        break;
                    case cv::GShape::GSCALAR:
                    {
                        cv::Scalar scalar;
                        if (pyopencv_to(item, scalar, ArgInfo("scalar", false)))
                        {
                            args.emplace_back(scalar);
                        }
                        else
                        {
                            util::throw_error(std::logic_error("Failed to obtain cv::Scalar"));
                        }
                        break;
                    }
                    case cv::GShape::GOPAQUE:
                        switch (info[i].kind)
                        {
                            case cv::detail::OpaqueKind::CV_SIZE:
                            {
                                cv::Size size;
                                if (pyopencv_to(item, size, ArgInfo("size", false)))
                                {
                                    args.emplace_back(cv::detail::OpaqueRef{std::move(size)});
                                }
                                else
                                {
                                    util::throw_error(std::logic_error("Failed to obtain cv::Size"));
                                }
                                break;
                            }
                            default:
                                GAPI_Assert(false);
                        }
                        break;
                    default:
                        GAPI_Assert(false);
                }
            }
            PyGILState_Release(gstate);
            return args;
    });

    return pyopencv_from(callback);
}

void inline bindInputs(const cv::GArg garg, PyObject* args, size_t idx)
{
    switch (garg.opaque_kind)
    {
        case cv::detail::OpaqueKind::CV_MAT:
        {
            PyTuple_SetItem(args, idx, pyopencv_from(garg.get<cv::Mat>()));
            break;
        }
        case cv::detail::OpaqueKind::CV_INT:
        {
            PyTuple_SetItem(args, idx, pyopencv_from(garg.get<int>()));
            break;
        }
        case cv::detail::OpaqueKind::CV_UNKNOWN:
        {
            PyTuple_SetItem(args, idx, garg.get<PyObject*>());
            break;
        }
        default:
            cv::util::throw_error(std::logic_error("Unsuported kernel input"));
    }
}

inline RMat::View asView(const Mat& m, RMat::View::DestroyCallback&& cb = nullptr) {
#if !defined(GAPI_STANDALONE)
    RMat::View::stepsT steps(m.dims);
    for (int i = 0; i < m.dims; i++) {
        steps[i] = m.step[i];
    }
    return RMat::View(cv::descr_of(m), m.data, steps, std::move(cb));
#else
    return RMat::View(cv::descr_of(m), m.data, m.step, std::move(cb));
#endif
}

class RMatAdapter : public RMat::Adapter {
    cv::Mat m_mat;
public:
    //const void* data() const { return m_mat.data; }
    RMatAdapter(cv::Mat m) : m_mat(m) {}
    virtual RMat::View access(RMat::Access) override { return asView(m_mat); }
    virtual cv::GMatDesc desc() const override { return cv::descr_of(m_mat); }
};

template <typename T>
void pyopencv_to_with_check(PyObject* from, T& to, const std::string& msg = "")
{
    if (!pyopencv_to(from, to, ArgInfo("", false)))
    {
        cv::util::throw_error(std::logic_error(msg));
    }
}

static cv::GRunArg extract_run_arg(const cv::GTypeInfo& info, PyObject* item)
{
	switch (info.shape)
	{
		case cv::GShape::GMAT:
        {
            // NB: In case streaming it can be IStreamSource or cv::Mat
            if (PyObject_TypeCheck(item,
                        reinterpret_cast<PyTypeObject*>(pyopencv_gapi_wip_IStreamSource_TypePtr)))
            {
                cv::gapi::wip::IStreamSource::Ptr source =
                    reinterpret_cast<pyopencv_gapi_wip_IStreamSource_t*>(item)->v;
                return source;
            }
            else
            {
                cv::Mat obj;
                pyopencv_to_with_check(item, obj, "Failed to obtain cv::Mat");
                return obj;
            }
            break;
        }
		case cv::GShape::GSCALAR:
        {
            cv::Scalar obj;
            pyopencv_to_with_check(item, obj, "Failed to obtain cv::Scalar");
            return obj;
            break;
        }
		default:
			util::throw_error(std::logic_error("Unsupported output shape"));
	}
}

static cv::GRunArgs extract_run_args(const cv::GTypesInfo& info, PyObject* py_args)
{
    cv::GRunArgs args;
    args.reserve(info.size());

    auto info_size = info.size();
    auto py_args_size = PyTuple_Size(py_args);
    if (info_size != py_args_size) {
        util::throw_error(std::logic_error("Size of tuple isn't equal to num of outputs"));
    }
    //GAPI_Assert(info_size == py_args_size);

    for (int i = 0; i < info_size; ++i)
    {
        PyObject* item = PyTuple_GetItem(py_args, i);
        args.emplace_back(extract_run_arg(info[i], item));
    }

    return args;
}

cv::GRunArgs runPythonKernel(PyObject* kernel,
                             const cv::GArgs& ins,
                             const cv::GTypesInfo& out_info) {
    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure();

    PyObject* args = PyTuple_New(ins.size());

    for (int i = 0; i < ins.size(); ++i)
    {
        bindInputs(ins[i], args, i);
    }

    PyObject* result = PyObject_CallObject(kernel, args);

    cv::GRunArgs outs;
    if (out_info.size() == 1)
    {
        outs = {extract_run_arg(out_info[0], result)};
    }
    else
    {
        outs = extract_run_args(out_info, result);
    }

    PyGILState_Release(gstate);

    return outs;
}

GMetaArgs empty_meta(const cv::GMetaArgs &meta, const cv::GArgs &args) {
    return {};
}

GMetaArgs python_meta(PyObject* outMeta, const cv::GMetaArgs &meta, const cv::GArgs &gargs) {
    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure();

    PyObject* args = PyTuple_New(meta.size());
    size_t idx = 0;
    for (auto&& m : meta)
    {
        switch (m.index())
        {
            case cv::GMetaArg::index_of<cv::GMatDesc>():
                PyTuple_SetItem(args, idx, pyopencv_from(cv::util::get<cv::GMatDesc>(m)));
                break;

            case cv::GMetaArg::index_of<cv::util::monostate>():
                PyTuple_SetItem(args, idx, gargs[idx].get<PyObject*>());
                break;

            default:
                util::throw_error(std::logic_error("Unsupported desc"));
        }
        ++idx;
    }

    PyObject* result = PyObject_CallObject(outMeta, args);
    bool is_tuple = PyTuple_Check(result);

    size_t size = is_tuple ? PyTuple_Size(result) : 1u;

    cv::GMetaArgs out_metas;
    out_metas.reserve(size);
    if (!is_tuple)
    {
        if (PyObject_TypeCheck(result,
                    reinterpret_cast<PyTypeObject*>(pyopencv_GMatDesc_TypePtr)))
        {
            out_metas.push_back(cv::GMetaArg{reinterpret_cast<pyopencv_GMatDesc_t*>(result)->v});
        }
    }
    else
    {
        util::throw_error(std::logic_error("Unsupported multiple metas"));
    }

    PyGILState_Release(gstate);

    return out_metas;
}

static PyObject* pyopencv_cv_gapi_kernels(PyObject* , PyObject* py_args, PyObject*)
{
    using namespace cv;
    gapi::GKernelPackage pkg;
    Py_ssize_t size = PyTuple_Size(py_args);
    for (int i = 0; i < size; ++i)
    {
        PyObject* pair   = PyTuple_GetItem(py_args, i);
        PyObject* kernel = PyTuple_GetItem(pair, 0);

        std::string tag;
        if (!pyopencv_to(PyTuple_GetItem(pair, 1), tag, ArgInfo("tag", false)))
        {
            // set error
        }

        Py_INCREF(kernel);
        gapi::python::GPythonFunctor f(tag.c_str(),
                                       empty_meta ,
                                       std::bind(runPythonKernel,
                                                 kernel,
                                                 std::placeholders::_1,
                                                 std::placeholders::_2));
        pkg.include(f);
    }
    return pyopencv_from(pkg);
}

static PyObject* pyopencv_cv_gapi_op(PyObject* , PyObject* py_args, PyObject*)
{
    using namespace cv;
    Py_ssize_t size = PyTuple_Size(py_args);
    std::string id;
    if (!pyopencv_to(PyTuple_GetItem(py_args, 0), id, ArgInfo("id", false)))
    {
        // set error
    }
    PyObject* outMeta = PyTuple_GetItem(py_args, 1);
    Py_INCREF(outMeta);

    cv::GArgs args;
    for (int i = 2; i < size; i++)
    {
        PyObject* item = PyTuple_GetItem(py_args, i);
        if (PyObject_TypeCheck(item,
                    reinterpret_cast<PyTypeObject*>(pyopencv_GMat_TypePtr)))
        {
            args.emplace_back(reinterpret_cast<pyopencv_GMat_t*>(item)->v);
        } 
        else if (PyObject_TypeCheck(item,
                           reinterpret_cast<PyTypeObject*>(pyopencv_GScalar_TypePtr)))
        {
            args.emplace_back(reinterpret_cast<pyopencv_GScalar_t*>(item)->v);
        }
        else
        {
            Py_INCREF(item);
            args.emplace_back(cv::GArg(item));
        }
    }

    cv::GKernel::M outMetaWrapper = std::bind(python_meta,
                                    outMeta,
                                    std::placeholders::_1,
                                    std::placeholders::_2);
    return pyopencv_from(cv::gapi::op(id, outMetaWrapper, std::move(args)));
}

#endif  // HAVE_OPENCV_GAPI
#endif  // OPENCV_GAPI_PYOPENCV_GAPI_HPP
