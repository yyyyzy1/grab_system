import sys

from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel, QListWidgetItem, QFileDialog, \
    QPushButton, QStackedWidget, QHBoxLayout, QSizePolicy,QListView
from PyQt5.uic import loadUi
from PyQt5.QtCore import pyqtSignal, QThread, QTimer, QDateTime,QMutex, QWaitCondition, QSize, QUrl
from FoundationPose.estimater import *
from FoundationPose.datareader import *
from PyQt5.QtGui import QImage, QPixmap,QStandardItemModel, QStandardItem,QIcon

from PIL import Image, ImageOps

class Window1(QWidget):
    def __init__(self):
        super().__init__()
        self.init_window1()
    def init_window1(self):
        self.setWindowTitle('Communication Check Content Screen')

        self.label_camera_message = QLabel('')
        self.label_robot_message = QLabel('')

        # 添加按钮用于打开文件选择对话框
        self.btn_check_camera = QPushButton('Check Communication between the Host Computer and the Camera')
        self.btn_check_camera.clicked.connect(self.open_available_camera)

        self.btn_check_robot = QPushButton('Check Communication between Upper Computer and Robot Arm')
        self.btn_check_robot.clicked.connect(self.open_available_robot)

        # 创建垂直布局并添加组件
        layout = QVBoxLayout()
        layout.addWidget(self.label_camera_message)
        layout.addWidget(self.label_robot_message)
        layout.addWidget(self.btn_check_camera)
        layout.addWidget(self.btn_check_robot)
        self.setLayout(layout)

    def open_available_camera(self):
        # TODO:检查相机通讯
        self.label_camera_message.setText('Camera Communication Successful!')


    def open_available_robot(self):
        # TODO:检查机械臂通讯
        self.label_robot_message.setText('Robotic Arm Communication Success!')


class Window2(QWidget):
    def __init__(self):
        super().__init__()
        self.init_window2()

    def init_window2(self):
        self.setWindowTitle('Camera Calibration Content Interface')

        self.label_check_camera_calibration = QLabel('')
        self.label_start_camera_calibration = QLabel('')

        # 添加按钮用于打开文件选择对话框
        self.btn_check_calibration = QPushButton('Check Calibration Accuracy')
        self.btn_check_calibration.clicked.connect(self.open_check_calibration)

        self.btn_start_calibration = QPushButton('Initial Calibration')
        self.btn_start_calibration.clicked.connect(self.open_start_calibration)


        # 创建垂直布局并添加组件
        layout = QVBoxLayout()
        layout.addWidget(self.label_check_camera_calibration)
        layout.addWidget(self.label_start_camera_calibration)
        layout.addWidget(self.btn_check_calibration)
        layout.addWidget(self.btn_start_calibration)

        self.setLayout(layout)

    def open_check_calibration(self):
        # TODO:检查相机通讯
        self.label_check_camera_calibration.setText('Accuracy is insufficient, please re-calibrate!')


    def open_start_calibration(self):
        # TODO:检查机械臂通讯
        self.label_start_camera_calibration.setText('Recalibration successful!')



class Window3(QWidget):
    log_message_signal_w3 = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.init_window3()

    def init_window3(self):
        self.setWindowTitle('Material Information Management Interface')

        # 创建 QListView 和模型来管理物料信息
        self.material_list_view = QListView()
        self.material_model = QStandardItemModel(self.material_list_view)
        self.material_list_view.setModel(self.material_model)

        # 连接双击信号到槽函数
        self.material_list_view.doubleClicked.connect(self.open_material_file)

        # 添加按钮用于打开文件选择对话框
        self.btn_select_file = QPushButton('Add Aaterial Information File')
        self.btn_select_file.clicked.connect(self.open_material_file_dialog)

        # 创建布局并添加组件
        layout = QVBoxLayout()
        layout.addWidget(QLabel('Available Material Information:'))
        layout.addWidget(self.material_list_view)  # 使用 QListView 显示物料信息
        layout.addWidget(self.btn_select_file)

        self.setLayout(layout)

        # 初始化列表，加载已有的物料信息
        self.load_material_info()

    def open_material_file_dialog(self):
        # 打开文件选择对话框
        options = QFileDialog.Options()
        add_material_file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Select the material file to import",
            "",
            "OBJ Files (*.obj);;PLY Files (*.ply)",  # 只允许选择 .obj 和 .ply 格式的文件
            options=options
        )

        if add_material_file_name:
            # 将新选中的物料信息添加到列表中
            self.add_material_info(add_material_file_name)

    def add_material_info(self, file_name):
        """将物料信息添加到 QListView 中，并更新数据库"""
        # 首先检查文件是否已经存在于模型中
        if self.is_material_exists(file_name):
            print(f"Material {file_name} already exists. Skipping addition.")
            return  # 如果已经存在，直接返回

        # 创建一个新条目并添加到模型中
        item = QStandardItem(file_name)
        self.material_model.appendRow(item)

        # 将新的物料文件路径保存到文件中
        with open('materials.txt', 'a') as f:
            f.write(file_name + '\n')

        self.log_message_signal_w3.emit(f"Added material file: {file_name}")

    def is_material_exists(self, file_name):
        """检查物料信息是否已经存在"""
        # 检查是否已经在 QListView 中存在
        for row in range(self.material_model.rowCount()):
            if self.material_model.item(row).text() == file_name:
                return True
        return False

    def load_material_info(self):
        """加载已有的物料信息（假设存储在本地文件中）"""
        try:
            with open('materials.txt', 'r') as f:
                materials = f.readlines()
                for material in materials:
                    material = material.strip()
                    if material and not self.is_material_exists(material):  # 避免重复添加
                        item = QStandardItem(material)
                        self.material_model.appendRow(item)
        except FileNotFoundError:
            # 如果文件不存在，可以忽略该错误
            pass

    def open_material_file(self, index):
        """使用指定的程序打开双击选择的物料文件"""
        file_path = self.material_model.itemFromIndex(index).text()

        # 移除可能的 'file://' 前缀，只保留本地文件路径
        if file_path.startswith('file://'):
            file_path = QUrl(file_path).toLocalFile()

        if os.path.exists(file_path):
            # 使用系统命令打开 MeshLab 或其他应用程序
            try:
                subprocess.Popen(['meshlab', file_path])  # 使用 meshlab 打开文件
            except FileNotFoundError:
                print("Could not find MeshLab. Please install it.")
        else:
            print(f"File {file_path} does not exist.")






class Window4(QWidget):

    mesh_file_selected = pyqtSignal(str)
    log_message_signal_w4 = pyqtSignal(str)


    def __init__(self):
        super().__init__()

        # 创建标签用于显示文字
        self.label_depth_text = QLabel('Depth Image')
        self.label_rgb_text = QLabel('RGB Image')
        self.label_pose_text = QLabel('Pose Visualization')
        #self.label_msg_text = QLabel('Run information output...')
        self.depth_picture = QLabel()
        self.rgb_picture = QLabel()
        self.pose_picture = QLabel()
        self.depth_picture.setScaledContents(True)
        self.rgb_picture.setScaledContents(True)
        self.pose_picture.setScaledContents(True)
        self.import_selected_material_file = QLabel('')
        self.btn_materia_import = QPushButton('Select the material to be grabbed')
        self.btn_materia_import.clicked.connect(self.import_material_file_dialog)

        # 创建水平布局并添加图片和文字
        layout1 = QVBoxLayout()
        layout1.addWidget(self.label_depth_text)
        layout1.addWidget(self.depth_picture)

        layout2 = QVBoxLayout()
        layout2.addWidget(self.label_rgb_text)
        layout2.addWidget(self.rgb_picture)

        layout3 = QVBoxLayout()
        layout3.addWidget(self.label_pose_text)
        layout3.addWidget(self.pose_picture)

        layout4 = QHBoxLayout()
        layout4.addLayout(layout1)
        layout4.addLayout(layout2)
        layout4.addLayout(layout3)
        layout6 = QVBoxLayout()
        layout6.addWidget(self.btn_materia_import)
        main_layout = QVBoxLayout()
        main_layout.addLayout(layout4)
        main_layout.addLayout(layout6)
        self.setLayout(main_layout)


    def import_material_file_dialog(self):
        # 打开文件选择对话框
        options = QFileDialog.Options()
        import_material_file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Select the material file to import",
            "",
            "OBJ Files (*.obj);;PLY Files (*.ply)",  # 只允许选择 .obj 和 .ply 格式的文件
            options=options
        )
        if import_material_file_name:
            print(type(import_material_file_name), import_material_file_name)
            #self.mesh_file = import_material_file_name
            self.import_selected_material_file.setText(
                'Select the material file to import：<br>{}'.format(import_material_file_name))
            self.mesh_file_selected.emit(import_material_file_name)
            # 发射日志消息信号，将文件名传递出去
            self.log_message_signal_w4.emit(f"Selected material file: {import_material_file_name}")


    def set_depth_image(self, depth_img):
        self.depth_picture.setPixmap(self.convert_np_to_pixmap(depth_img))

    def set_rgb_image(self, rgb_img):
        self.rgb_picture.setPixmap(self.convert_np_to_pixmap(rgb_img))

    def set_pose_image(self, pose_img):
        self.pose_picture.setPixmap(self.convert_np_to_pixmap(pose_img))

    def convert_np_to_pixmap(self, img_np):
        # 将 numpy 数组转换为 QPixmap
        height, width, channel = img_np.shape
        bytes_per_line = 3 * width
        q_image = QImage(img_np.data, width, height, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(q_image)



class DetThread(QThread):

    depth_signal = pyqtSignal(np.ndarray)
    rgb_signal = pyqtSignal(np.ndarray)
    pose_signal = pyqtSignal(np.ndarray)
    log_signal = pyqtSignal(str)

    def __init__(self):
        super(DetThread, self).__init__()
        self.mesh_file = None
        self.stop_flag = False  # 停止标志
        self.mutex = QMutex()  # 互斥锁
        self.wait_condition = QWaitCondition()  # 条件变量
        self.paused = False  # 暂停标志

    def set_mesh_file(self, mesh_file):
        self.mesh_file = mesh_file  # 通过该方法设置 mesh_file

    def apply_pseudocolor_pil(self, depth_image):
        # 将深度图像归一化到 0-255 范围，并转换为 8-bit 格式
        depth_min = np.min(depth_image)
        depth_max = np.max(depth_image)

        # 将深度图像转换为 8-bit 图像
        depth_image_8bit = ((depth_image - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)

        # 将 numpy 数组转换为 PIL 图像
        depth_image_pil = Image.fromarray(depth_image_8bit)

        # 使用自动对比度增强图像
        depth_image_normalized = ImageOps.autocontrast(depth_image_pil)

        # 使用调色板映射，类似于伪彩色
        depth_image_colored = depth_image_normalized.convert('P')

        # 自定义伪彩色调色板 (示例使用蓝-红渐变)
        palette = []
        for i in range(256):
            # 蓝-红渐变的调色板: 从蓝色 (0, 0, 255) 到红色 (255, 0, 0)
            r = int(255 * (i / 255.0))
            g = 0
            b = int(255 * ((255 - i) / 255.0))
            palette.extend((r, g, b))

        # 为图像添加调色板
        depth_image_colored.putpalette(palette)

        # 转换为 RGB 图像
        depth_image_rgb = depth_image_colored.convert('RGB')
        depth_image_rgb_np = np.array(depth_image_rgb)

        return depth_image_rgb_np

    def stop(self):
        self.stop_flag = True  # 设置停止标志

    def run(self):
        self.stop_flag = False
            # 检查是否需要暂停
        mesh = trimesh.load(self.mesh_file)
        self.log_signal.emit("Mesh loaded successfully!")
        debug = 2
        debug_dir = f'{code_dir}/debug'
        os.system(f'rm -rf {debug_dir}/* && mkdir -p {debug_dir}/track_vis {debug_dir}/ob_in_cam')

        to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
        bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)
        ##########init###########
        self.log_signal.emit("Model initialization...")

        if self.stop_flag:
            return

        scorer = ScorePredictor()
        refiner = PoseRefinePredictor()
        glctx = dr.RasterizeCudaContext()
        est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer,
                             refiner=refiner, debug_dir=debug_dir, debug=debug, glctx=glctx)
        logging.info("estimator initialization done")
        ####################################
        self.log_signal.emit("Model initialization done!")
        reader = YcbineoatReader(video_dir="../FoundationPose/demo_data/mustard0", shorter_side=None, zfar=np.inf)
        self.log_signal.emit("Start detection...")
        for i in range(len(reader.color_files)):


            logging.info(f'i:{i}')
            color = reader.get_color(i)
            depth = reader.get_depth(i)
            depth_image_rgb = self.apply_pseudocolor_pil(depth)

            self.depth_signal.emit(depth_image_rgb)
            self.rgb_signal.emit(color)

            if self.stop_flag:
                self.log_signal.emit("Thread stopped!")
                break
            if i == 0:
                mask = reader.get_mask(0).astype(bool)
                start = time.time()
                pose = est.register(K=reader.K, rgb=color, depth=depth, ob_mask=mask, iteration=5)
                end = time.time()
                print("time_register", end - start)
                if debug >= 3:
                    m = mesh.copy()
                    m.apply_transform(pose)
                    m.export(f'{debug_dir}/model_tf.obj')
                    xyz_map = depth2xyzmap(depth, reader.K)
                    valid = depth >= 0.001
                    pcd = toOpen3dCloud(xyz_map[valid], color[valid])
                    o3d.io.write_point_cloud(f'{debug_dir}/scene_complete.ply', pcd)
                self.log_signal.emit("The attitude estimation of the first grasped object has been completed!")

            else:

                # mask = reader.get_mask(i).astype(bool)
                # pose = est.register_test(K=reader.K, rgb=color, depth=depth, ob_mask=mask,
                #                          iteration=5)
                if self.stop_flag:
                    self.log_signal.emit("Thread stopped!")
                    break
                pose = est.track_one(rgb=color, depth=depth, K=reader.K, iteration=2)
                self.log_signal.emit("The attitude estimation of the {} grasped object has been completed!".format(i))

            os.makedirs(f'{debug_dir}/ob_in_cam', exist_ok=True)
            np.savetxt(f'{debug_dir}/ob_in_cam/{reader.id_strs[i]}.txt', pose.reshape(4, 4))

            if debug >= 1:
                center_pose = pose @ np.linalg.inv(to_origin)
                vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
                vis = draw_xyz_axis(vis, ob_in_cam=center_pose, scale=0.1, K=reader.K, thickness=3, transparency=0,
                                    is_input_rgb=True)
                self.pose_signal.emit(vis)
                #cv2.imshow('1', vis[..., ::-1])
                # cv2.waitKey(1)

            if debug >= 2:
                os.makedirs(f'{debug_dir}/track_vis', exist_ok=True)
                imageio.imwrite(f'{debug_dir}/track_vis/{reader.id_strs[i]}.png', vis)








class MyWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # 加载UI文件
        loadUi('qt/main_window.ui', self)

        window_size = self.stackedWidget.size()
        self.w1 = Window1()
        self.w2 = Window2()
        self.w3 = Window3()
        self.w4 = Window4()
        self.stackedWidget.addWidget(self.w1)
        self.stackedWidget.addWidget(self.w2)
        self.stackedWidget.addWidget(self.w3)
        self.stackedWidget.addWidget(self.w4)

        self.w3.log_message_signal_w3.connect(self.add_log_message)

        self.w4.mesh_file_selected.connect(self.handle_mesh_file_selected)
        self.w4.log_message_signal_w4.connect(self.add_log_message)

        # 其他初始化操作
        self.setWindowTitle("Material Grabbing AI System")
        self.btn_commu.clicked.connect(self.show_commu_page)
        self.btn_cam.clicked.connect(self.show_cam_demarcate_page)
        self.btn_material_model_import.clicked.connect(self.show_material_model_page)
        self.btn_material_grab.clicked.connect(self.show_material_grab_page)
        self.pushButton_start.setIcon(QIcon('/home/yan/code/foundation_qt/qt/icons/run.png'))
        self.pushButton_start.setIconSize(QSize(32, 22))
        self.pushButton_start.setToolTip('Run')
        self.pushButton_stop.setIcon(QIcon('/home/yan/code/foundation_qt/qt/icons/stop.png'))
        self.pushButton_stop.setIconSize(QSize(32, 22))
        self.pushButton_stop.setToolTip('Stop')
        self.pushButton_check.setIcon(QIcon('/home/yan/code/foundation_qt/qt/icons/check.png'))
        self.pushButton_check.setIconSize(QSize(32, 22))
        self.pushButton_check.setToolTip('Check')
        self.pushButton_save.setIcon(QIcon('/home/yan/code/foundation_qt/qt/icons/save.png'))
        self.pushButton_save.setIconSize(QSize(32, 22))
        self.pushButton_save.setToolTip('Save')
        self.btn_commu.setIcon(QIcon('/home/yan/code/foundation_qt/qt/icons/commu.png'))
        self.btn_commu.setIconSize(QSize(32, 22))
        self.btn_commu.setToolTip('Communication')
        self.btn_cam.setIcon(QIcon('/home/yan/code/foundation_qt/qt/icons/camera.png'))
        self.btn_cam.setIconSize(QSize(32, 22))
        self.btn_cam.setToolTip('Camera')
        self.btn_material_model_import.setIcon(QIcon('/home/yan/code/foundation_qt/qt/icons/database.png'))
        self.btn_material_model_import.setIconSize(QSize(32, 22))
        self.btn_material_model_import.setToolTip('Database')
        self.btn_material_grab.setIcon(QIcon('/home/yan/code/foundation_qt/qt/icons/grab.png'))
        self.btn_material_grab.setIconSize(QSize(32, 22))
        self.btn_material_grab.setToolTip('Grab')

        self.pushButton_start.clicked.connect(self.check_and_start_thread)
        self.pushButton_stop.clicked.connect(self.stop_thread)

        self.list_view = self.findChild(QListView, 'listView')

    # 创建数据模型
        self.model = QStandardItemModel(self)

    # 将模型设置到 QListView
        self.list_view.setModel(self.model)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_time)  # 连接定时器的超时信号到更新时间的槽函数
        self.timer.start(1000)
        self.update_time()  # 初始化时立即更新一次时间



    def update_time(self):
        # 获取当前日期和时间
        current_datetime = QDateTime.currentDateTime()
        # 格式化日期和时间为 'yyyy-MM-dd HH:mm:ss' 形式
        datetime_text = current_datetime.toString('yyyy-MM-dd HH:mm:ss')
        # 更新 QLabel 的文本
        self.top_time_label2.setText(datetime_text)



    def check_and_start_thread(self):
        # 检查当前页面是否是w4
        if self.stackedWidget.currentWidget() == self.w4:
            self.start_det_thread()  # 只有在 w4 页面时触发线程
        else:
            pass

    def handle_mesh_file_selected(self, mesh_file):
        # 当文件被选择时保存文件名
        self.mesh_file = mesh_file

    def start_det_thread(self):
        if hasattr(self, 'mesh_file') and self.mesh_file:
            self.det_thread = DetThread()
            self.det_thread.depth_signal.connect(self.w4.set_depth_image)
            self.det_thread.rgb_signal.connect(self.w4.set_rgb_image)
            self.det_thread.pose_signal.connect(self.w4.set_pose_image)
            self.det_thread.set_mesh_file(self.mesh_file)  # 将文件名传递给线程
            self.det_thread.log_signal.connect(self.add_log_message)
            self.det_thread.start()
        else:
            print("No mesh file selected.")

    def stop_thread(self):
        if hasattr(self, 'det_thread') and self.det_thread.isRunning():
            self.det_thread.stop()
            self.det_thread.wait()
            print("Thread stopped.")

    def add_log_message(self, message):
        """将消息添加到列表视图中"""

        # 最大保留的消息数量
        MAX_LOG_COUNT = 50

        # 插入新消息
        item = QStandardItem(message)
        self.model.appendRow(item)

        # 如果消息数量超过了最大限制，则删除最前面的几条
        if self.model.rowCount() > MAX_LOG_COUNT:
            rows_to_remove = self.model.rowCount() - MAX_LOG_COUNT
            self.model.removeRows(0, rows_to_remove)

        # 确保列表视图滚动到底部
        self.list_view.scrollToBottom()



    def show_commu_page(self):
        self.stackedWidget.setCurrentIndex(2)
    def show_cam_demarcate_page(self):

        self.stackedWidget.setCurrentIndex(3)

    def show_material_model_page(self):
        self.stackedWidget.setCurrentIndex(4)

    def show_material_grab_page(self):
        self.stackedWidget.setCurrentIndex(5)






if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyWindow()
    window.showMaximized()
    window.add_log_message("Welcome!")

    window.show()
    sys.exit(app.exec_())
