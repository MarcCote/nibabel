import copy
import numbers
import numpy as np
import collections
from warnings import warn

from nibabel.affines import apply_affine

from .array_sequence import ArraySequence


def is_data_dict(obj):
    """ True if `obj` seems to implement the :class:`DataDict` API """
    return hasattr(obj, 'store')


def is_lazy_dict(obj):
    """ True if `obj` seems to implement the :class:`LazyDict` API """
    return is_data_dict(obj) and callable(obj.store.values()[0])


class SliceableDataDict(collections.MutableMapping):
    """ Dictionary for which key access can do slicing on the values.

    This container behaves like a standard dictionary but extends key access to
    allow keys for key access to be indices slicing into the contained ndarray
    values.
    """
    def __init__(self, *args, **kwargs):
        self.store = dict()
        # Use update to set the keys.
        if len(args) != 1:
            self.update(dict(*args, **kwargs))
            return
        if args[0] is None:
            return
        if isinstance(args[0], SliceableDataDict):
            self.update(**args[0])
        else:
            self.update(dict(*args, **kwargs))

    def __getitem__(self, key):
        try:
            return self.store[key]
        except (KeyError, TypeError):
            pass  # Maybe it is an integer or a slicing object

        # Try to interpret key as an index/slice for every data element, in
        # which case we perform (maybe advanced) indexing on every element of
        # the dictionnary.
        idx = key
        new_dict = type(self)(None)
        try:
            for k, v in self.items():
                new_dict[k] = v[idx]
        except TypeError:
            pass
        else:
            return new_dict

        # Key was not a valid index/slice after all.
        return self.store[key]  # Will raise the proper error.

    def __delitem__(self, key):
        del self.store[key]

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)


class PerArrayDict(SliceableDataDict):
    """ Dictionary for which key access can do slicing on the values.

    This container behaves like a standard dictionary but extends key access to
    allow keys for key access to be indices slicing into the contained ndarray
    values.  The elements must also be ndarrays.

    In addition, it makes sure the amount of data contained in those ndarrays
    matches the number of streamlines given at the instantiation of this
    dictionary.
    """
    def __init__(self, n_elements, *args, **kwargs):
        self.n_elements = n_elements
        super(PerArrayDict, self).__init__(*args, **kwargs)

    def __setitem__(self, key, value):
        value = np.asarray(list(value))

        if value.ndim == 1 and value.dtype != object:
            # Reshape without copy
            value.shape = ((len(value), 1))

        if value.ndim != 2:
            raise ValueError("data_per_streamline must be a 2D array.")

        # We make sure there is the right amount of values
        if self.n_elements is not None and len(value) != self.n_elements:
            msg = ("The number of values ({0}) should match n_elements "
                   "({1}).").format(len(value), self.n_elements)
            raise ValueError(msg)

        self.store[key] = value


class LazyDict(SliceableDataDict):
    """ Dictionary of generator functions.

    This container behaves like an dictionary but it makes sure its elements
    are callable objects and assumed to be generator function yielding values.
    When getting the element associated to a given key, the element (i.e. a
    generator function) is first called before being returned.
    """
    def __init__(self, *args, **kwargs):
        if len(args) == 1 and isinstance(args[0], LazyDict):
            # Copy the generator functions.
            self.store = dict()
            self.update(**args[0].store)
            return
        super(LazyDict, self).__init__(*args, **kwargs)

    def __getitem__(self, key):
        return self.store[key]()

    def __setitem__(self, key, value):
        if value is not None and not callable(value):
            raise TypeError("`value` must be a generator function or None.")
        self.store[key] = value


class TractogramItem(object):
    """ Class containing information about one streamline.

    :class:`TractogramItem` objects have three main properties: `streamline`,
    `data_for_streamline`, and `data_for_points`.

    Parameters
    ----------
    streamline : ndarray shape (N, 3)
        Points of this streamline represented as an ndarray of shape (N, 3)
        where N is the number of points.
    data_for_streamline : dict
        Dictionary containing some data associated to this particular
        streamline. Each key `k` is mapped to a ndarray of shape (Pk,), where
        `Pt` is the dimension of the data associated with key `k`.
    data_for_points : dict
        Dictionary containing some data associated to each point of this
        particular streamline. Each key `k` is mapped to a ndarray of
        shape (Nt, Mk), where `Nt` is the number of points of this streamline
        and `Mk` is the dimension of the data associated with key `k`.
    """
    def __init__(self, streamline, data_for_streamline, data_for_points):
        self.streamline = np.asarray(streamline)
        self.data_for_streamline = data_for_streamline
        self.data_for_points = data_for_points

    def __iter__(self):
        return iter(self.streamline)

    def __len__(self):
        return len(self.streamline)


class Tractogram(object):
    """ Container for streamlines and their data information.

    Streamlines of a tractogram can be in any coordinate system of your
    choice as long as you provide the correct `affine_to_rasmm` matrix, at
    construction time, that brings the streamlines back to *RAS+*, *mm* space,
    where the coordinates (0,0,0) corresponds to the center of the voxel
    (opposed to a corner).

    Attributes
    ----------
    streamlines : :class:`ArraySequence` object
        Sequence of $T$ streamlines. Each streamline is an ndarray of
        shape ($N_t$, 3) where $N_t$ is the number of points of
        streamline $t$.
    data_per_streamline : dict of 2D arrays
        Dictionary where the items are (str, 2D array).
        Each key represents an information $i$ to be kept along side every
        streamline, and its associated value is a 2D array of shape
        ($T$, $P_i$) where $T$ is the number of streamlines and $P_i$ is
        the number scalar values to store for that particular information $i$.
    data_per_point : dict of :class:`ArraySequence` objects
        Dictionary where the items are (str, :class:`ArraySequence`).
        Each key represents an information $i$ to be kept along side every
        point of every streamline, and its associated value is an iterable
        of ndarrays of shape ($N_t$, $M_i$) where $N_t$ is the number of
        points for a particular streamline $t$ and $M_i$ is the number
        scalar values to store for that particular information $i$.
    """
    def __init__(self, streamlines=None,
                 data_per_streamline=None,
                 data_per_point=None,
                 affine_to_rasmm=np.eye(4)):
        """
        Parameters
        ----------
        streamlines : iterable of ndarrays or :class:`ArraySequence`, optional
            Sequence of $T$ streamlines. Each streamline is an ndarray of
            shape ($N_t$, 3) where $N_t$ is the number of points of
            streamline $t$.
        data_per_streamline : dict of iterable of ndarrays, optional
            Dictionary where the items are (str, iterable).
            Each key represents an information $i$ to be kept along side every
            streamline, and its associated value is an iterable of ndarrays of
            shape ($P_i$,) where $P_i$ is the number scalar values to store
            for that particular information $i$.
        data_per_point : dict of iterable of ndarrays, optional
            Dictionary where the items are (str, iterable).
            Each key represents an information $i$ to be kept along side every
            point of every streamline, and its associated value is an iterable
            of ndarrays of shape ($N_t$, $M_i$) where $N_t$ is the number of
            points for a particular streamline $t$ and $M_i$ is the number
            scalar values to store for that particular information $i$.
        affine_to_rasmm : ndarray of shape (4, 4), optional
            Transformation matrix that brings the streamlines contained in
            this tractogram to *RAS+* and *mm* space where coordinate (0,0,0)
            refers to the center of the voxel. By default, the streamlines
            are assumed to be already in *RAS+* and *mm* space.
        """
        self.streamlines = streamlines
        self.data_per_streamline = data_per_streamline
        self.data_per_point = data_per_point
        self._affine_to_rasmm = affine_to_rasmm

    @property
    def streamlines(self):
        return self._streamlines

    @streamlines.setter
    def streamlines(self, value):
        self._streamlines = ArraySequence(value)

    @property
    def data_per_streamline(self):
        return self._data_per_streamline

    @data_per_streamline.setter
    def data_per_streamline(self, value):
        self._data_per_streamline = DataPerStreamlineDict(self, value)

    @property
    def data_per_point(self):
        return self._data_per_point

    @data_per_point.setter
    def data_per_point(self, value):
        self._data_per_point = DataPerPointDict(self, value)

    def get_affine_to_rasmm(self):
        """ Returns the affine bringing this tractogram to RAS+mm. """
        return self._affine_to_rasmm.copy()

    def __iter__(self):
        for i in range(len(self.streamlines)):
            yield self[i]

    def __getitem__(self, idx):
        pts = self.streamlines[idx]

        data_per_streamline = {}
        for key in self.data_per_streamline:
            data_per_streamline[key] = self.data_per_streamline[key][idx]

        data_per_point = {}
        for key in self.data_per_point:
            data_per_point[key] = self.data_per_point[key][idx]

        if isinstance(idx, (numbers.Integral, np.integer)):
            return TractogramItem(pts, data_per_streamline, data_per_point)

        return Tractogram(pts, data_per_streamline, data_per_point)

    def __len__(self):
        return len(self.streamlines)

    def copy(self):
        """ Returns a copy of this :class:`Tractogram` object. """
        return copy.deepcopy(self)

    def apply_affine(self, affine, lazy=False):
        """ Applies an affine transformation on the points of each streamline.

        If `lazy` is not specified, this is performed *in-place*.

        Parameters
        ----------
        affine : ndarray of shape (4, 4)
            Transformation that will be applied to every streamline.
        lazy : {False, True}, optional
            If True, streamlines are *not* transformed in-place and a
            :class:`LazyTractogram` object is returned. Otherwise, streamlines
            are modified in-place.

        Returns
        -------
        tractogram : :class:`Tractogram` or :class:`LazyTractogram` object
            Tractogram where the streamlines have been transformed according
            to the given affine transformation. If the `lazy` option is true,
            it returns a :class:`LazyTractogram` object, otherwise it returns a
            reference to this :class:`Tractogram` object with updated
            streamlines.
        """
        if lazy:
            lazy_tractogram = LazyTractogram.from_tractogram(self)
            return lazy_tractogram.apply_affine(affine)

        if len(self.streamlines) == 0:
            return self

        BUFFER_SIZE = 10000000  # About 128 Mb since pts shape is 3.
        for start in range(0, len(self.streamlines._data), BUFFER_SIZE):
            end = start + BUFFER_SIZE
            pts = self.streamlines._data[start:end]
            self.streamlines._data[start:end] = apply_affine(affine, pts)

        # Update the affine that brings back the streamlines to RASmm.
        self._affine_to_rasmm = np.dot(self._affine_to_rasmm,
                                       np.linalg.inv(affine))

        return self

    def to_world(self, lazy=False):
        """ Brings the streamlines to world space (i.e. RAS+ and mm).

        If `lazy` is not specified, this is performed *in-place*.

        Parameters
        ----------
        lazy : {False, True}, optional
            If True, streamlines are *not* transformed in-place and a
            :class:`LazyTractogram` object is returned. Otherwise, streamlines
            are modified in-place.

        Returns
        -------
        tractogram : :class:`Tractogram` or :class:`LazyTractogram` object
            Tractogram where the streamlines have been sent to world space.
            If the `lazy` option is true, it returns a :class:`LazyTractogram`
            object, otherwise it returns a reference to this
            :class:`Tractogram` object with updated streamlines.
        """
        return self.apply_affine(self._affine_to_rasmm, lazy=lazy)


class LazyTractogram(Tractogram):
    """ Lazy container for streamlines and their data information.

    This container behaves lazily as it uses generator functions to manage
    streamlines and their data information. This container is thus memory
    friendly since it doesn't require having all those data loaded in memory.

    Streamlines of a lazy tractogram can be in any coordinate system of your
    choice as long as you provide the correct `affine_to_rasmm` matrix, at
    construction time, that brings the streamlines back to *RAS+*, *mm* space,
    where the coordinates (0,0,0) corresponds to the center of the voxel
    (opposed to a corner).

    Attributes
    ----------
    streamlines : generator function
        Generator function yielding streamlines. Each streamline is an
        ndarray of shape ($N_t$, 3) where $N_t$ is the number of points of
        streamline $t$.
    data_per_streamline : :class:`LazyDict` object
        Dictionary where the items are (str, instantiated generator).
        Each key represents an information $i$ to be kept along side every
        streamline, and its associated value is a generator function
        yielding that information via ndarrays of shape ($P_i$,) where
        $P_i$ is the number scalar values to store for that particular
        information $i$.
    data_per_point : :class:`LazyDict` object
        Dictionary where the items are (str, instantiated generator).
        Each key represents an information $i$ to be kept along side every
        point of every streamline, and its associated value is a generator
        function yielding that information via ndarrays of shape
        ($N_t$, $M_i$) where $N_t$ is the number of points for a particular
        streamline $t$ and $M_i$ is the number scalar values to store for
        that particular information $i$.

    Notes
    -----
    LazyTractogram objects do not support indexing currently.
    """
    def __init__(self, streamlines=None,
                 data_per_streamline=None,
                 data_per_point=None,
                 affine_to_rasmm=np.eye(4)):
        """
        Parameters
        ----------
        streamlines : generator function, optional
            Generator function yielding streamlines. Each streamline is an
            ndarray of shape ($N_t$, 3) where $N_t$ is the number of points of
            streamline $t$.
        data_per_streamline : dict of generator functions, optional
            Dictionary where the items are (str, generator function).
            Each key represents an information $i$ to be kept along side every
            streamline, and its associated value is a generator function
            yielding that information via ndarrays of shape ($P_i$,) where
            $P_i$ is the number scalar values to store for that particular
            information $i$.
        data_per_point : dict of generator functions, optional
            Dictionary where the items are (str, generator function).
            Each key represents an information $i$ to be kept along side every
            point of every streamline, and its associated value is a generator
            function yielding that information via ndarrays of shape
            ($N_t$, $M_i$) where $N_t$ is the number of points for a particular
            streamline $t$ and $M_i$ is the number scalar values to store for
            that particular information $i$.
        affine_to_rasmm : ndarray of shape (4, 4)
            Transformation matrix that brings the streamlines contained in
            this tractogram to *RAS+* and *mm* space where coordinate (0,0,0)
            refers to the center of the voxel.
        """
        super(LazyTractogram, self).__init__(streamlines,
                                             data_per_streamline,
                                             data_per_point,
                                             affine_to_rasmm)
        self._nb_streamlines = None
        self._data = None
        self._affine_to_apply = np.eye(4)

    @classmethod
    def from_tractogram(cls, tractogram):
        """ Creates a :class:`LazyTractogram` object from a :class:`Tractogram` object.

        Parameters
        ----------
        tractogram : :class:`Tractgogram` object
            Tractogram from which to create a :class:`LazyTractogram` object.

        Returns
        -------
        lazy_tractogram : :class:`LazyTractogram` object
            New lazy tractogram.
        """
        lazy_tractogram = cls(lambda: tractogram.streamlines.copy())

        # Set data_per_streamline using data_func
        def _gen(key):
            return lambda: iter(tractogram.data_per_streamline[key])

        for k in tractogram.data_per_streamline:
            lazy_tractogram._data_per_streamline[k] = _gen(k)

        # Set data_per_point using data_func
        def _gen(key):
            return lambda: iter(tractogram.data_per_point[key])

        for k in tractogram.data_per_point:
            lazy_tractogram._data_per_point[k] = _gen(k)

        lazy_tractogram._nb_streamlines = len(tractogram)
        lazy_tractogram._affine_to_rasmm = tractogram.get_affine_to_rasmm()
        return lazy_tractogram

    @classmethod
    def create_from(cls, data_func):
        """ Creates an instance from a generator function.

        The generator function must yield :class:`TractogramItem` objects.

        Parameters
        ----------
        data_func : generator function yielding :class:`TractogramItem` objects
            Generator function that whenever it is called starts yielding
            :class:`TractogramItem` objects that will be used to instantiate a
            :class:`LazyTractogram`.

        Returns
        -------
        lazy_tractogram : :class:`LazyTractogram` object
            New lazy tractogram.
        """
        if not callable(data_func):
            raise TypeError("`data_func` must be a generator function.")

        lazy_tractogram = cls()
        lazy_tractogram._data = data_func

        try:
            first_item = next(data_func())

            # Set data_per_streamline using data_func
            def _gen(key):
                return lambda: (t.data_for_streamline[key] for t in data_func())

            data_per_streamline_keys = first_item.data_for_streamline.keys()
            for k in data_per_streamline_keys:
                lazy_tractogram._data_per_streamline[k] = _gen(k)

            # Set data_per_point using data_func
            def _gen(key):
                return lambda: (t.data_for_points[key] for t in data_func())

            data_per_point_keys = first_item.data_for_points.keys()
            for k in data_per_point_keys:
                lazy_tractogram._data_per_point[k] = _gen(k)

        except StopIteration:
            pass

        return lazy_tractogram

    @property
    def streamlines(self):
        streamlines_gen = iter([])
        if self._streamlines is not None:
            streamlines_gen = self._streamlines()
        elif self._data is not None:
            streamlines_gen = (t.streamline for t in self._data())

        # Check if we need to apply an affine.
        if not np.allclose(self._affine_to_apply, np.eye(4)):
            def _apply_affine():
                for s in streamlines_gen:
                    yield apply_affine(self._affine_to_apply, s)

            return _apply_affine()

        return streamlines_gen

    @streamlines.setter
    def streamlines(self, value):
        if value is not None and not callable(value):
            raise TypeError("`streamlines` must be a generator function.")

        self._streamlines = value

    @property
    def data_per_streamline(self):
        return self._data_per_streamline

    @data_per_streamline.setter
    def data_per_streamline(self, value):
        self._data_per_streamline = LazyDict(self, value)

    @property
    def data_per_point(self):
        return self._data_per_point

    @data_per_point.setter
    def data_per_point(self, value):
        self._data_per_point = LazyDict(self, value)

    @property
    def data(self):
        if self._data is not None:
            return self._data()

        def _gen_data():
            data_per_streamline_generators = {}
            for k, v in self.data_per_streamline.items():
                data_per_streamline_generators[k] = iter(v)

            data_per_point_generators = {}
            for k, v in self.data_per_point.items():
                data_per_point_generators[k] = iter(v)

            for s in self.streamlines:
                data_for_streamline = {}
                for k, v in data_per_streamline_generators.items():
                    data_for_streamline[k] = next(v)

                data_for_points = {}
                for k, v in data_per_point_generators.items():
                    data_for_points[k] = next(v)

                yield TractogramItem(s, data_for_streamline, data_for_points)

        return _gen_data()

    def __getitem__(self, idx):
        raise NotImplementedError('`LazyTractogram` does not support indexing.')

    def __iter__(self):
        count = 0
        for tractogram_item in self.data:
            yield tractogram_item
            count += 1

        # Keep how many streamlines there are in this tractogram.
        self._nb_streamlines = count

    def __len__(self):
        # Check if we know how many streamlines there are.
        if self._nb_streamlines is None:
            warn("Number of streamlines will be determined manually by looping"
                 " through the streamlines. If you know the actual number of"
                 " streamlines, you might want to set it beforehand via"
                 " `self.header.nb_streamlines`."
                 " Note this will consume any generators used to create this"
                 " `LazyTractogram` object.", Warning)
            # Count the number of streamlines.
            self._nb_streamlines = sum(1 for _ in self.streamlines)

        return self._nb_streamlines

    def copy(self):
        """ Returns a copy of this :class:`LazyTractogram` object. """
        tractogram = LazyTractogram(self._streamlines,
                                    self._data_per_streamline,
                                    self._data_per_point)
        tractogram._nb_streamlines = self._nb_streamlines
        tractogram._data = self._data
        tractogram._affine_to_apply = self._affine_to_apply.copy()
        return tractogram

    def apply_affine(self, affine, lazy=True):
        """ Applies an affine transformation to the streamlines.

        The transformation given by the `affine` matrix is applied after any
        other pending transformations to the streamline points.

        Parameters
        ----------
        affine : 2D array (4,4)
            Transformation matrix that will be applied on each streamline.
        lazy : True, optional
            Should always be True for :class:`LazyTractogram` object. Doing
            otherwise will raise a ValueError.

        Returns
        -------
        lazy_tractogram : :class:`LazyTractogram` object
            A copy of this :class:`LazyTractogram` instance but with a
            transformation to be applied on the streamlines.
        """
        if not lazy:
            msg = "LazyTractogram only supports lazy transformations."
            raise ValueError(msg)

        tractogram = self.copy()  # New instance.

        # Update the affine that will be applied when returning streamlines.
        tractogram._affine_to_apply = np.dot(affine, self._affine_to_apply)

        # Update the affine that brings back the streamlines to RASmm.
        tractogram._affine_to_rasmm = np.dot(self._affine_to_rasmm,
                                             np.linalg.inv(affine))
        return tractogram

    def to_world(self, lazy=True):
        """ Brings the streamlines to world space (i.e. RAS+ and mm).

        The transformation is applied after any other pending transformations
        to the streamline points.

        Parameters
        ----------
        lazy : True, optional
            Should always be True for :class:`LazyTractogram` object. Doing
            otherwise will raise a ValueError.

        Returns
        -------
        lazy_tractogram : :class:`LazyTractogram` object
            A copy of this :class:`LazyTractogram` instance but with a
            transformation to be applied on the streamlines.
        """
        return self.apply_affine(self._affine_to_rasmm, lazy=lazy)
