import pytest
import xcompact3d_toolbox as x3d
import xcompact3d_toolbox.mesh


@pytest.fixture
def possible_mesh():
    return {
        9,
        11,
        13,
        17,
        19,
        21,
        25,
        31,
        33,
        37,
        41,
        49,
        51,
        55,
        61,
        65,
        73,
        81,
        91,
        97,
        101,
        109,
        121,
        129,
        145,
        151,
        161,
        163,
        181,
        193,
        201,
        217,
        241,
        251,
        257,
        271,
        289,
        301,
        321,
        325,
        361,
        385,
        401,
        433,
        451,
        481,
        487,
        501,
        513,
        541,
        577,
        601,
        641,
        649,
        721,
        751,
        769,
        801,
        811,
        865,
        901,
        961,
        973,
        1001,
        1025,
        1081,
        1153,
        1201,
        1251,
        1281,
        1297,
        1351,
        1441,
        1459,
        1501,
        1537,
        1601,
        1621,
        1729,
        1801,
        1921,
        1945,
        2001,
        2049,
        2161,
        2251,
        2305,
        2401,
        2431,
        2501,
        2561,
        2593,
        2701,
        2881,
        2917,
        3001,
        3073,
        3201,
        3241,
        3457,
        3601,
        3751,
        3841,
        3889,
        4001,
        4051,
        4097,
        4321,
        4375,
        4501,
        4609,
        4801,
        4861,
        5001,
        5121,
        5185,
        5401,
        5761,
        5833,
        6001,
        6145,
        6251,
        6401,
        6481,
        6751,
        6913,
        7201,
        7291,
        7501,
        7681,
        7777,
        8001,
        8101,
        8193,
        8641,
        8749,
        9001,
    }

@pytest.fixture
def possible_mesh_periodic(possible_mesh):
    return {i - 1 for i in possible_mesh}

def test_coordinate_grid_size_value(possible_mesh):
    coordinate = x3d.mesh.Coordinate(is_periodic = False)
    assert set(coordinate.get_possible_grid_size_values(0, 9002)) == possible_mesh

def test_coordinate_periodic_grid_size_value(possible_mesh_periodic):
    coordinate = x3d.mesh.Coordinate(is_periodic = True)
    assert set(coordinate.get_possible_grid_size_values(0, 9002)) == possible_mesh_periodic

@pytest.fixture
def coordinate():
    return x3d.mesh.Coordinate()

@pytest.mark.parametrize("length", [1.0, 10.0, 100.0])
@pytest.mark.parametrize("grid_size", [101, 201])
@pytest.mark.parametrize("is_periodic", [True, False])
def test_coordinate_properties(coordinate, length, grid_size, is_periodic):
    if is_periodic: grid_size -= 1
    sub_grid_size = grid_size if is_periodic else grid_size - 1
    delta = length / grid_size if is_periodic else length / (grid_size - 1)
    coordinate.is_periodic = is_periodic
    coordinate.grid_size = grid_size
    coordinate.length = length

    assert (sub_grid_size, delta) == (coordinate._sub_grid_size, coordinate.delta)

    new_length = coordinate.length / 2.0
    coordinate.delta /= 2.0

    assert new_length == coordinate.length

