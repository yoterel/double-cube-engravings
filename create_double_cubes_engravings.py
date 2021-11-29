import vtk
import sys
import cv2 as cv
import numpy as np
import triangle
import os
from PIL import Image
from pathlib import Path


def load_file(file):
    reader = vtk.vtkOBJReader()
    input_poly = vtk.vtkPolyData()
    reader.SetFileName(str(file))
    reader.Update()
    input_poly.ShallowCopy(reader.GetOutput())
    return input_poly


def export_decimated_poly(poly, file_name):
    with open(file_name, 'w+') as f:
        for i in range(poly.GetNumberOfPoints()):
            p = [0, 0, 0]
            poly.GetPoint(i, p)
            # vertices.append(p)
            f.write("v %f %f %f\n" % (p[0], p[1], p[2]))
        for i in range(poly.GetNumberOfPolys()):
            face = poly.GetCell(i).GetPointIds()
            # faces.append([face.GetId(0), face.GetId(1), face.GetId(2)])
            f.write("f %d %d %d\n" % (face.GetId(0) + 1, face.GetId(1) + 1, face.GetId(2) + 1))


def vtk_render(object, axes_loc=[0, 0, 0]):
    colors = vtk.vtkNamedColors()

    transform = vtk.vtkTransform()
    transform.Translate(axes_loc[0], axes_loc[1], axes_loc[2])

    axes = vtk.vtkAxesActor()
    #  The axes are positioned with a user transform
    axes.SetUserTransform(transform)

    # properties of the axes labels can be set as follows
    # this sets the x axis label to red
    # axes.GetXAxisCaptionActor2D().GetCaptionTextProperty().SetColor(colors.GetColor3d('Red'));

    # the actual text of the axis label can be changed:
    # axes->SetXAxisLabelText('test');


    cylinderMapper = vtk.vtkPolyDataMapper()
    cylinderMapper.SetInputConnection(object.GetOutputPort())
    # The actor is a grouping mechanism: besides the geometry (mapper), it
    # also has a property, transformation matrix, and/or texture map.
    # Here we set its color and rotate it -22.5 degrees.
    cylinderActor = vtk.vtkActor()
    cylinderActor.SetMapper(cylinderMapper)
    cylinderActor.GetProperty().SetColor(colors.GetColor3d("Tomato"))
    # cylinderActor.RotateX(30.0)
    # cylinderActor.RotateY(-45.0)
    # Create the graphics structure. The renderer renders into the render
    # window. The render window interactor captures mouse events and will
    # perform appropriate camera or actor manipulation depending on the
    # nature of the events.

    # Set the background color.
    bkg = map(lambda x: x / 255.0, [26, 51, 102, 255])
    colors.SetColor("BkgColor", *bkg)
    ren = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)

    # Add the actors to the renderer, set the background and size
    ren.AddActor(cylinderActor)
    ren.AddActor(axes)
    ren.SetBackground(colors.GetColor3d("BkgColor"))
    renWin.SetSize(300, 300)
    renWin.SetWindowName('CylinderExample')

    # This allows the interactor to initalize itself. It has to be
    # called before an event loop.
    iren.Initialize()

    # We'll zoom in a little by accessing the camera and invoking a "Zoom"
    # method on it.
    # ren.ResetCamera()
    # ren.GetActiveCamera().Zoom(1.5)
    renWin.Render()

    # Start the event loop.
    iren.Start()


def merge_cube_and_pattern(cube, extrusion):
    cube_tri = vtk.vtkTriangleFilter()
    cube_tri.SetInputData(cube)
    # define a y-rotation and translation transformer (moves and rotates engraving on cube face)
    theta = np.random.uniform(0, 360)
    translate_x = np.random.uniform(-0.9, 0.9)
    translate_y = np.random.uniform(-0.9, 0.9)
    trans = vtk.vtkTransform()
    trans.Translate(translate_x, 0, translate_y)
    rot = vtk.vtkTransform()
    rot.RotateY(theta)
    rot.Concatenate(trans)
    transformer = vtk.vtkTransformPolyDataFilter()
    # apply transformation to extruded pattern
    transformer.SetInputData(extrusion)
    transformer.SetTransform(rot)
    transformer.Update()
    # vtk_render(transformer, [0, 0.1, 0])
    extrude_tri = vtk.vtkTriangleFilter()
    extrude_tri.SetInputData(transformer.GetOutput())
    # apply boolean difference opration to obtain engraving from extruded pattern + cube
    booleanOperation = vtk.vtkBooleanOperationPolyDataFilter()
    booleanOperation.SetOperationToDifference()
    # booleanOperation.SetOperationToUnion()
    # booleanOperation.SetOperationToIntersection()
    booleanOperation.SetInputConnection(0, cube_tri.GetOutputPort())
    booleanOperation.SetInputConnection(1, extrude_tri.GetOutputPort())
    booleanOperation.Update()
    # vtk_render(booleanOperation, [0, 0.1, 0])
    return booleanOperation


def make_cube_two_sided(files, output):
    """
    engraves cube with two patterns
    :param files:
    :param output:
    :return:
    """
    other_dice_side = np.random.randint(0, 5)
    cube = vtk.vtkCubeSource()
    # centered cube with sides = 3
    cube.SetXLength(3)
    cube.SetYLength(3)
    cube.SetZLength(3)
    # cube center is moved, and y side is now at 0.1 (extrude is between 0 and 0.2 in y)
    cube.SetCenter(0, -1.4, 0)
    cube.Update()
    extrude1 = load_file(files[0])
    extrude2 = load_file(files[1])
    result = merge_cube_and_pattern(cube.GetOutput(), extrude1)
    # translate->rotate->translate structure for second engraving
    dice_rot = vtk.vtkTransform()
    dice_trans_forward = vtk.vtkTransform()
    dice_trans_backward = vtk.vtkTransform()
    dice_trans_forward.Translate(0, 1.4, 0)
    dice_trans_backward.Translate(0, -1.4, 0)
    # dice_rot.RotateX(90)
    if other_dice_side == 0:
        theta = np.array(90)
        dice_rot.RotateX(theta)
    elif other_dice_side == 1:
        theta = np.array(180)
        dice_rot.RotateX(theta)
    elif other_dice_side == 2:
        theta = np.array(270)
        dice_rot.RotateX(theta)
    elif other_dice_side == 3:
        theta = np.array(90)
        dice_rot.RotateZ(theta)
    elif other_dice_side == 4:
        theta = np.array(-90)
        dice_rot.RotateZ(theta)
    # apply transformation to engraved cube
    dice_rot.Concatenate(dice_trans_forward)
    dice_trans_backward.Concatenate(dice_rot)
    transformer = vtk.vtkTransformPolyDataFilter()
    transformer.SetInputData(result.GetOutput())
    transformer.SetTransform(dice_trans_backward)
    transformer.Update()
    #########
    result = merge_cube_and_pattern(transformer.GetOutput(), extrude2)
    # apply random rotation to result in all axis to avoid aligned dataset
    theta = np.random.uniform(0, 360, size=3)
    rot_x = vtk.vtkTransform()
    rot_x.RotateX(theta[0])
    rot_y = vtk.vtkTransform()
    rot_y.RotateY(theta[1])
    rot_z = vtk.vtkTransform()
    rot_z.RotateZ(theta[2])
    rot_x.Concatenate(rot_y)
    rot_x.Concatenate(rot_z)
    transformer.SetInputData(result.GetOutput())
    transformer.SetTransform(rot_x)
    transformer.Update()
    # use result only if number of faces meets criterion
    num_of_faces = transformer.GetOutput().GetNumberOfPolys()
    featureEdges = vtk.vtkFeatureEdges()
    featureEdges.FeatureEdgesOff()
    featureEdges.BoundaryEdgesOff()
    featureEdges.NonManifoldEdgesOn()
    featureEdges.SetInputData(transformer.GetOutput())
    featureEdges.Update()
    non_manifold_edges = featureEdges.GetOutput().GetNumberOfCells()
    if non_manifold_edges == 0:
        if 100 < num_of_faces < 1600:
            export_decimated_poly(transformer.GetOutput(), output)
            return True
        else:
            print("rejected mesh because num of faces={}:".format(num_of_faces))
    else:
        print("rejected mesh because non manifold edges={}:".format(non_manifold_edges))

    return False


def make_cube(file, output):
    for _ in range(5):
        cube = vtk.vtkCubeSource()
        cube.SetXLength(3)
        cube.SetYLength(3)
        cube.SetZLength(3)
        cube.SetCenter(0, -1.4, 0)
        cube.Update()
        input1 = cube.GetOutput()
        cube_tri = vtk.vtkTriangleFilter()
        cube_tri.SetInputData(input1)

        bat = load_file(file)
        theta = np.random.uniform(0, 360)
        translate_x = np.random.uniform(-0.9, 0.9)
        translate_y = np.random.uniform(-0.9, 0.9)
        trans = vtk.vtkTransform()
        trans.Translate(translate_x, 0, translate_y)
        rot = vtk.vtkTransform()
        rot.RotateY(theta)
        rot.Concatenate(trans)
        transformer = vtk.vtkTransformPolyDataFilter()
        transformer.SetInputData(bat)
        transformer.SetTransform(rot)
        transformer.Update()
        bat_tri = vtk.vtkTriangleFilter()
        bat_tri.SetInputData(transformer.GetOutput())
        booleanOperation = vtk.vtkBooleanOperationPolyDataFilter()
        booleanOperation.SetOperationToDifference()
        # booleanOperation.SetOperationToUnion()
        # booleanOperation.SetOperationToIntersection()
        booleanOperation.SetInputConnection(0, cube_tri.GetOutputPort())
        booleanOperation.SetInputConnection(1, bat_tri.GetOutputPort())
        booleanOperation.Update()
        theta = np.random.uniform(0, 360, size=3)
        rot_x = vtk.vtkTransform()
        rot_x.RotateX(theta[0])
        rot_y = vtk.vtkTransform()
        rot_y.RotateY(theta[1])
        rot_z = vtk.vtkTransform()
        rot_z.RotateZ(theta[2])

        rot_x.Concatenate(rot_y)
        rot_x.Concatenate(rot_z)

        transformer.SetInputData(booleanOperation.GetOutput())
        transformer.SetTransform(rot_x)
        transformer.Update()
        num_of_faces = transformer.GetOutput().GetNumberOfPolys()

        featureEdges = vtk.vtkFeatureEdges()
        featureEdges.FeatureEdgesOff()
        featureEdges.BoundaryEdgesOff()
        featureEdges.NonManifoldEdgesOn()
        featureEdges.SetInputData(transformer.GetOutput())
        featureEdges.Update()
        edges = featureEdges.GetOutput().GetNumberOfCells()
        if 100 < num_of_faces < 650 and edges == 0:
            export_decimated_poly(transformer.GetOutput(), output)
            return True
    print("skipping {}".format(file))
    return False


def cube_folder(input_extrudes, output_folder, cubes_per_extrude, classes_per_cube, dataset_size):
    files = []
    for file in input_extrudes.glob("*.obj"):
        files.append(file)
    all_classes = np.unique(sorted([x.name.split('-')[0] for x in files]))
    if classes_per_cube == 1:
        for file in files:
            class_name = file.name.split('-')[0]
            class_dir = Path(output_folder, class_name)
            class_dir.mkdir(parents=True, exist_ok=True)
            for i in range(cubes_per_extrude):
                dst = Path(class_dir, "{}_{:04d}.obj".format(file.stem, i))
                if not make_cube(file, dst):
                    break
    elif classes_per_cube == 2:
        gt_file = Path(output_folder, "labels.csv")
        cur_size = 0
        gt_dict = {}
        while cur_size < dataset_size:
            selected_files = np.random.choice(np.array(files), size=2, replace=False)
            dst = Path(output_folder, "{:06d}.obj".format(cur_size))
            if not dst.is_file():
                print(selected_files[0].name, selected_files[1].name)
                if make_cube_two_sided(selected_files, dst):
                    cur_size += 1
                    print(cur_size)
                    selected_classes = np.array([x.stem.split("-")[0] for x in selected_files])
                    label = np.isin(all_classes, selected_classes)
                    gt_dict[dst] = label
                    with open(gt_file, "a+") as f:
                        val = np.array2string(label.astype(int), max_line_width=1000, separator=",")
                        f.write("{},{}\n".format(dst, val))

            else:
                cur_size += 1
                print(cur_size)



def scale_contour(contour):
    max_val = np.max(contour)
    min_val = np.min(contour)
    contour = (contour - min_val) / (max_val - min_val)
    contour[:,:, 0] -= (np.max(contour[:, :, 0]) + np.min(contour[:, :, 0])) / 2
    contour[:,:, 1] -= (np.max(contour[:, :, 1]) + np.min(contour[:, :, 1])) / 2
    return contour


def export_obj(vertices, faces, export_path, name):
        with open('%s/%s.obj' % (export_path, name.lower()), 'w+') as f:
            for vertex in vertices:
                f.write("v %f %f %f\n" % (vertex[0], vertex[1], vertex[2]))
            for face_id in range(len(faces) - 1):
                f.write("f %d %d %d\n" % (faces[face_id][0] + 1, faces[face_id][1] + 1, faces[face_id][2] + 1))
            f.write("f %d %d %d" % (faces[-1][0] + 1, faces[-1][1] + 1, faces[-1][2] + 1))


def extrude_segments(segments, height=.5):
    """
    extrudes a triangulation "segments" in the y direction by height
    :param segments: the triangulation (vertices and triangles)
    :param height: the height to extrude with
    :return: the new triangulation (Vertices and triangles)
    """
    faces_bottom = segments["triangles"]
    vertices = segments["vertices"]
    vertices_count = len(vertices)
    faces_top = faces_bottom.copy()
    faces_top += len(vertices)
    faces_top = np.flip(faces_top, 1)
    vertices_bottom = np.zeros([vertices_count,3])
    vertices_bottom [:, 0] = vertices[:, 0]
    vertices_bottom [:, 2] = vertices[:, 1]
    vertices_top = vertices_bottom.copy()
    vertices_top[:, 1] = height
    vertices = np.concatenate((vertices_bottom, vertices_top), 0)
    side_faces = np.zeros([vertices_count * 2, 3], dtype=np.int32)
    for i in range(vertices_count):
        side_faces[i*2, :] = [i, (i + 1) % vertices_count, (i + 1) % vertices_count + vertices_count]
        side_faces[i*2 + 1, :] = [i, (i + 1) % vertices_count + vertices_count, i + vertices_count]
    faces = np.concatenate((faces_bottom, faces_top, side_faces), 0)
    return vertices, faces


def read_image(file):
    image = np.array(Image.open(file), dtype=np.uint8)
    image = np.expand_dims(image, 2)
    return np.repeat(image, 3, axis=2)


def image_to_mesh(input_image, output_path, file_name):
    if Path(output_path, file_name + ".obj").is_file():
        return
    img = read_image(input_image)
    if img is None:
        print("Image {} not found".format(input_image))
        sys.exit(1)
    img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    thresh = np.zeros(img.shape[:2], dtype=np.uint8)
    thresh[img > 0.8] = 1
    # ret, thresh = cv.threshold(img, 127, 255, 0)
    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_TC89_L1)
    if len(contours) == 0:
        print(input_image)
        print('error')
        return
    contour = contours[0]
    if len(contours) > 1:
        for c in contours:
            if c.shape[0] > contour.shape[0]:
                contour = c
    epsilon = 0.001 * cv.arcLength(contour, True)
    contour = cv.approxPolyDP(contour, epsilon, True)
    v = {"vertices": np.array([]), "segments": np.array([])}
    overhead = 0
    contour = scale_contour(contour)
    contour = list(map(lambda x: [x[0][0],  x[0][1]], contour))
    contour_segments = []
    for s in range(len(contour)):
        contour_segments.append([s + overhead, ((s+1) % (len(contour))) + overhead])
    overhead += len(contour)
    v["vertices"] = np.array(contour)
    v["segments"] = np.array(contour_segments)
    segments = triangle.triangulate(v, 'p')
    vertices, faces = extrude_segments(segments)
    export_obj(vertices, faces, output_path, file_name)
    # print("done")


def convert_image(input_image, output_path, file_name):
    im = Image.open(input_image)
    background = Image.new("RGB", im.size, (255, 255, 255))
    background.paste(im)
    background.save("%s/%s.jpg" % (output_path, file_name), 'JPEG')


def imgs_to_meshes(input_path, export_path):
    # classes = ['apple', 'bat', 'bell', 'bone', 'brick', 'camel', 'car', 'chopper', 'elephant', 'fork',
    #            'guitar', 'hammer', 'heart', 'horseshoe', 'key', 'lmfish', 'octopus', 'shoe', 'spoon', 'tree',
    #            'turtle', 'watch']
    # classes = ['apple']
    for file in input_path.glob("*.gif"):
        image_to_mesh(file, export_path, file.stem)
        # if file.stem.split("-")[0].lower() in classes:
        #     image_to_mesh(file, export_path, file.stem)


def jpg_folder(input_path, export_path):
    for root, _, files in os.walk(input_path):
        for file in files:
            file_name, file_extension = os.path.splitext(file)
            if file_extension == '.gif':
                convert_image(os.path.join(root, file), export_path, file_name)

def parse_arguments():
    parser = argparse.ArgumentParser(description='double-cube-engravings')
    parser.add_argument('MPEG7', type=str, help='Path to MPEG7 dataset containg the .gif files')
    parser.add_argument('extrudes', type=str, help='Path to create extruded obj files in.')
    parser.add_argument('cubes', type=str, help='Path to create final output in')
	parser.add_argument('nclass_pcube', type=int, default=2, choices=[1, 2], help='number of engravings per cube')
	parser.add_argument('dataset_size', type=int, default=1e5, help='size of dataset (used only if nclass_pcube=2)')
	parser.add_argument('ncubes_pextrude', type=int, default=10, help='number of extrudes per cube (only used if nclass_pcube=1)')
	
    args = parser.parse_args()
	extrudes.mkdir(exist_ok=True, parents=True)
    cubes_path.mkdir(exist_ok=True, parents=True)
	args.MPEG7 = Path(MPEG7)
	args.extrudes = Path(extrudes)
	args.cubes = Path(cubes)
    return args
	

if __name__ == "__main__":
    """
    creates double cubes engravings from MPEG7 shape dataset
    download MPEG7 from here : https://dabi.temple.edu/external/shape/MPEG7/dataset.html
    """
	args = parse_arguments()
    imgs_to_meshes(args.MPEG7, args.extrudes)
    cube_folder(args.extrudes, args.cubes,
                cubes_per_extrude=args.ncubes_pextrude,
                classes_per_cube=args.nclass_pcube,
				args.dataset_size)



