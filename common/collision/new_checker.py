import sys
import pathlib
import copy
import math
import numpy
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

import geometry_msgs.msg
from common.scenario.obstacle import Obstacle

class SATCollisionChecker(object):
    def check_collision(self, ego: Obstacle, obstacle: Obstacle):
        egoPos = copy.deepcopy(ego.pos)
        obsPos = copy.deepcopy(obstacle.pos)
        egoPoly = copy.deepcopy(ego.polygon)
        obsPoly = copy.deepcopy(obstacle.polygon)

        for i in egoPoly:
            i += egoPos
        for i in obsPoly:
            i += obsPos

        edges1 = self.vertices_to_edges(egoPoly)
        edges2 = self.vertices_to_edges(obsPoly)
        edges1 = numpy.delete(edges1,-1,0)
        edges2 = numpy.delete(edges2,-1,0)
        edges1 = numpy.append(edges1,edges2)
        axes = []
        
        for i in range(numpy.shape(edges1)[0]):
            axes = numpy.append(axes, self._get_orthogonal(edges1[i]))

        for i in range(numpy.shape(axes)[0]):
            projection1 = self._project(egoPoly, axes[i])
            projection2 = self._project(obsPoly, axes[i])

            if(not(self._is_overlapping(projection1,projection2))):
                return False
        return True

    def construct_rectancle(self, centre_x, centre_y, yaw, length, width, margin_lon, margin_lat):
        p1 = geometry_msgs.msg.Point32()
        p2 = geometry_msgs.msg.Point32()
        p3 = geometry_msgs.msg.Point32()
        p4 = geometry_msgs.msg.Point32()
        egoPolyangle = geometry_msgs.msg.Polygon()

        bottom_x = -length*(0.5 + margin_lon)
        top_x = length*(0.5 + margin_lon)
        left_y = -width*(0.5 + margin_lat)
        right_y = width*(0.5 + margin_lat)
        p1.x = bottom_x
        p1.y = left_y

        p2.x = top_x
        p2.y = left_y

        p3.x = top_x
        p3.y = right_y

        p4.x = bottom_x
        p4.y = right_y

        egoPolyangle.points = []
        egoPolyangle.points.extend([p1,p2,p3,p4])

        egoPolyangle = self._rotate_and_translate_egoPoly(egoPolyangle, centre_x, centre_y, yaw)
        egoPolyangle.points.append(egoPolyangle.points[0])
        return egoPolyangle

    def construct_straight_bumper(self,centre_x, centre_y, yaw, length, width, margin):
        p1 = geometry_msgs.msg.Point32()
        p2 = geometry_msgs.msg.Point32()
        p3 = geometry_msgs.msg.Point32()
        p4 = geometry_msgs.msg.Point32()
        egoPolyangle = geometry_msgs.msg.Polygon()

        bottom_x = -1.5
        top_x = 1.5 + length
        left_y = -width / 2 - margin
        right_y = width / 2 + margin

        p1.x = bottom_x
        p1.y = left_y

        p2.x = top_x
        p2.y = left_y

        p3.x = top_x
        p3.y = right_y

        p4.x = bottom_x
        p4.y = right_y

        egoPolyangle.points.extend([p1,p2,p3,p4])
        egoPolyangle = self._rotate_and_translate_egoPoly(egoPolyangle, centre_x, centre_y, yaw)
        egoPolyangle.points.append(egoPolyangle.points[0])
        return egoPolyangle

    def  remove_top_layer(poly):
        new_polygon = geometry_msgs.msg.Polygon()
        num_points = poly.points.size()

        for i in range(num_points/2):
            new_polygon.points.append(poly.points[i])      
        return new_polygon
        
    def vertices_to_edges(self, poly):
        num_vertices = len(poly)
        edges = []
        for i in range(num_vertices-1):
            edges = numpy.append(edges,self._get_edge_direction(poly[i],poly[i+1]))
        return edges

    def _get_edge_direction(self, p0, p1):
        direction = geometry_msgs.msg.Vector3()

        direction.x = p1[0] - p0[0]
        direction.y = p1[1] - p0[1]

        return direction

    def _get_orthogonal(self, vec):
        orthogonal = geometry_msgs.msg.Vector3()

        orthogonal.x = vec.y
        orthogonal.y = -vec.x

        return orthogonal

    def _normalize(self, vec):
        normalized = geometry_msgs.msg.Vector3()

        norm = (vec.x**2+vec.y**2)**0.5
        normalized.x = vec.x/norm
        normalized.y = vec.y/norm
        
        return normalized

    def _project(self, poly, axis):
        products = []
        for i in range(numpy.shape(poly)[0]):
            products.append(self._get_dot_product(poly[i], axis))

        minimum = min(products)
        maximum = max(products)
        return [minimum, maximum]

    def _get_dot_product(self, point, axis):
        return point[0]*axis.x + point[1]*axis.y

    def _is_overlapping(self,projection1, projection2):
        if projection1[0]<projection2[0]:
            return True if projection2[0] - projection1[1] < 0 else False
        else:
            return True if projection1[0] - projection2[1] < 0 else False

    def _rotate_and_translate_egoPoly(egoPoly, centre_x, centre_y, yaw):
        new_point = geometry_msgs.msg.Point32()
        rotated_egoPoly = geometry_msgs.msg.Polygon()

        for point in egoPoly.points:
            x_orig = point.x
            y_orig = point.y
            new_point.x = x_orig * math.cos(yaw) - y_orig * math.sin(yaw)
            new_point.y = x_orig * math.sin(yaw) + y_orig * math.cos(yaw)
            new_point.x += centre_x
            new_point.y += centre_y
            rotated_egoPoly.points = []
            rotated_egoPoly.points.append(new_point)

        return rotated_egoPoly
