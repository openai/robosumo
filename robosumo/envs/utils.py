import colorsys
import numpy as np
import os
import xml.etree.ElementTree as ET


def cart2pol(vec):
    """Convert a cartesian 2D vector to polar coordinates."""
    r = np.sqrt(vec[0]**2 + vec[1]**2)
    theta = np.arctan2(vec[1], vec[0])
    return np.asarray([r, theta])


def get_distinct_colors(n=2):
    """Source: https://stackoverflow.com/a/876872."""
    HSV_tuples = [(x * 1. / n, .5, .5) for x in range(n)]
    RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
    return RGB_tuples


def _set_class(root, prop, name):
    if root is None:
        return
    if root.tag == prop:
        root.set('class', name)
    for child in list(root):
        _set_class(child, prop, name)


def _add_prefix(root, prop, prefix, force_set=False):
    if root is None:
        return
    root_prop_val = root.get(prop)
    if root_prop_val is not None:
        root.set(prop, prefix + '/' + root_prop_val)
    elif force_set:
        root.set(prop, prefix + '/' + 'anon' + str(np.random.randint(1, 1e10)))
    for child in list(root):
        _add_prefix(child, prop, prefix, force_set)


def _tuple_to_str(tp):
    return " ".join(map(str, tp))


def construct_scene(scene_xml_path, agent_xml_paths,
                    agent_densities=None,
                    agent_scopes=None,
                    init_poses=None,
                    rgb=None,
                    tatami_size=None):
    """Construct an XML that represents a MuJoCo scene for sumo."""
    n_agents = len(agent_xml_paths)
    assert n_agents == 2, "Only 2-agent sumo is currently supported."

    scene = ET.parse(scene_xml_path)
    scene_root = scene.getroot()
    scene_default = scene_root.find('default')
    scene_body = scene_root.find('worldbody')
    scene_actuator = None
    scene_sensors = None

    # Set tatami size if specified
    if tatami_size is not None:
        for geom in scene_body.findall('geom'):
            if geom.get('name') == 'tatami':
                size = tatami_size + 0.3
                geom.set('size', "{size:.2f} {size:.2f} 0.25".format(size=size))
            if geom.get('name') == 'topborder':
                fromto = \
                    "-{size:.2f} {size:.2f} 0.5  {size:.2f} {size:.2f} 0.5" \
                    .format(size=tatami_size)
                geom.set('fromto', fromto)
            if geom.get('name') == 'rightborder':
                fromto = \
                    "{size:.2f} -{size:.2f} 0.5  {size:.2f} {size:.2f} 0.5" \
                    .format(size=tatami_size)
                geom.set('fromto', fromto)
            if geom.get('name') == 'bottomborder':
                fromto = \
                    "-{size:.2f} -{size:.2f} 0.5  {size:.2f} -{size:.2f} 0.5" \
                    .format(size=tatami_size)
                geom.set('fromto', fromto)
            if geom.get('name') == 'leftborder':
                fromto = \
                    "-{size:.2f} -{size:.2f} 0.5  -{size:.2f} {size:.2f} 0.5" \
                    .format(size=tatami_size)
                geom.set('fromto', fromto)

    # Resolve colors
    if rgb is None:
        rgb = get_distinct_colors(n_agents)
    else:
        assert len(rgb) == n_agents, "Each agent must have a color."
    RGBA_tuples = list(map(lambda x: _tuple_to_str(x + (1,)), rgb))

    # Resolve densities
    if agent_densities is None:
        agent_densities = [10.0] * n_agents

    # Resolve scopes
    if agent_scopes is None:
        agent_scopes = ['agent' + str(i) for i in range(n_agents)]
    else:
        assert len(agent_scopes) == n_agents, "Each agent must have a scope."

    # Resolve initial positions
    if init_poses is None:
        r, phi, z = 1.5, 0., .75
        delta = (2. * np.pi) / n_agents
        init_poses = []
        for i in range(n_agents):
            angle = phi + i * delta
            x, y = r * np.cos(angle), r * np.sin(angle)
            init_poses.append((x, y, z))

    # Build agent XMLs
    for i in range(n_agents):
        agent_xml = ET.parse(agent_xml_paths[i])
        agent_default = ET.SubElement(
            scene_default, 'default',
            attrib={'class': agent_scopes[i]}
        )

        # Set defaults
        rgba = RGBA_tuples[i]
        density = str(agent_densities[i])
        default_set = False
        for child in list(agent_xml.find('default')):
            if child.tag == 'geom':
                child.set('rgba', rgba)
                child.set('density', density)
                default_set = True
            agent_default.append(child)
        if not default_set:
            agent_geom = ET.SubElement(
                agent_default, 'geom',
                attrib={
                    'density': density,
                    'contype': '1',
                    'conaffinity': '1',
                    'rgba': rgba,
                }
            )

        # Build agent body
        agent_body = agent_xml.find('body')
        # set initial position
        agent_body.set('pos', _tuple_to_str(init_poses[i]))
        # add class to all geoms
        _set_class(agent_body, 'geom', agent_scopes[i])
        # add prefix to all names, important to map joints
        _add_prefix(agent_body, 'name', agent_scopes[i], force_set=True)
        # add aggent body to xml
        scene_body.append(agent_body)

        # Build agent actuators
        agent_actuator = agent_xml.find('actuator')
        # add class and prefix to all motor joints
        _add_prefix(agent_actuator, 'joint', agent_scopes[i])
        _add_prefix(agent_actuator, 'name', agent_scopes[i])
        _set_class(agent_actuator, 'motor', agent_scopes[i])
        # add actuator
        if scene_actuator is None:
            scene_root.append(agent_actuator)
            scene_actuator = scene_root.find('actuator')
        else:
            for motor in list(agent_actuator):
                scene_actuator.append(motor)

        # Build agent sensors
        agent_sensors = agent_xml.find('sensor')
        # add same prefix to all sensors
        _add_prefix(agent_sensors, 'joint', agent_scopes[i])
        _add_prefix(agent_sensors, 'name', agent_scopes[i])
        if scene_sensors is None:
            scene_root.append(agent_sensors)
            scene_sensors = scene_root.find('sensor')
        else:
            for sensor in list(agent_sensors):
                scene_sensors.append(sensor)

    return scene
