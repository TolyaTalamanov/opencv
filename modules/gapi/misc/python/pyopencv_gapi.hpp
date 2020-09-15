using gapi_GKernelPackage = cv::gapi::GKernelPackage;
using GProtoInputArgs  = GIOProtoArgs<In_Tag>;
using GProtoOutputArgs = GIOProtoArgs<Out_Tag>;

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


PyObject* from_grunarg(const GRunArg& v)
{
    switch (v.index())
    {
        case GRunArg::index_of<cv::Mat>():
        {
            const auto& m = util::get<cv::Mat>(v);
            return pyopencv_from(m);
        }

        default:
            GAPI_Assert(false);
    }
    return NULL;
}

template<>
PyObject* pyopencv_from(const GRunArgs& value)
{
    int i, n = (int)value.size();
    PyObject* seq = PyList_New(n);
    for( i = 0; i < n; i++ )
    {
        PyObject* item = from_grunarg(value[i]);
        if(!item)
            break;
        PyList_SetItem(seq, i, item);
    }
    if( i < n )
    {
        Py_DECREF(seq);
        return 0;
    }
    return seq;
}

template <typename T>
static PyObject* extract_proto_args(PyObject* py_args, PyObject* kw)
{
    using namespace cv;

    GProtoArgs args;
    Py_ssize_t size = PyTuple_Size(py_args);
    for (int i = 0; i < size; ++i) {
        PyObject* item = PyTuple_GetItem(py_args, i);
        if (PyObject_TypeCheck(item, (PyTypeObject*)pyopencv_GScalar_TypePtr)) {
            args.emplace_back(((pyopencv_GScalar_t*)item)->v);
        } else if (PyObject_TypeCheck(item, (PyTypeObject*)pyopencv_GMat_TypePtr)) {
            args.emplace_back(((pyopencv_GMat_t*)item)->v);
        } else {
            PyErr_SetString(PyExc_TypeError, "cv.GIn() supports only cv.GMat and cv.GScalar");
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

static PyObject* pyopencv_cv_gin(PyObject* , PyObject* py_args, PyObject* kw)
{
    using namespace cv;

    GRunArgs args;
    Py_ssize_t size = PyTuple_Size(py_args);
    for (int i = 0; i < size; ++i) {
        PyObject* item = PyTuple_GetItem(py_args, i);
        if (PyTuple_Check(item)) {
            cv::Scalar s;
            pyopencv_to(item, s, ArgInfo("scalar", i));
            args.emplace_back(s);
        } else if (PyArray_Check(item)) {
            cv::Mat m;
            pyopencv_to(item, m, ArgInfo("mat", i));
            args.emplace_back(m);
        }
    }

    return pyopencv_from_generic_vec(args);
}

static PyObject* pyopencv_cv_gout(PyObject* o, PyObject* py_args, PyObject* kw)
{
    return pyopencv_cv_gin(o, py_args, kw);
}
