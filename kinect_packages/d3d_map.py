def d3d_map(self):
    # http://archive.petercollingridge.co.uk/book/export/html/460
    self.show_color_pixel = False

    pygame.init()
    factor = 2
    zoom_factor = 2
    self.screen = pygame.display.set_mode((192 * factor * 2, 108 * factor * 2 + 192 * factor),
                                          pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.RESIZABLE, 32)
    self.color_surface = pygame.Surface((1920, 1080), 0, 32)
    self.depth_surface = pygame.Surface((self._kinect.depth_frame_desc.Width, self._kinect.depth_frame_desc.Height), 0,
                                        24)
    self.node_surface = pygame.Surface((1000 * zoom_factor, 1000 * zoom_factor), 0, 32)
    frame = 0
    got_frame = False
    begin_time = time.time()

    self.debug_time = {"mapping": 0, "transforming": 0, "displaying": 0, "new_mapping": 0}
    self.status = {"offset": [630 * zoom_factor, -440 * zoom_factor], "scaling_factor": 1, "rotate": [0, 0, 0]}

    while True:
        if got_frame:
            frame += 1
            if frame == 1:
                begin_time = time.time()
        if self._kinect.has_new_color_frame():
            color_frame = self._kinect.get_last_color_frame()
            self.draw_color_frame(color_frame, self.color_surface)
            got_frame = True
            passed_time = time.time() - begin_time
            print("frame", frame, round(passed_time, 2), round(frame / passed_time, 2))
        if self._kinect.has_new_depth_frame():
            depth_frame_og = self._kinect.get_last_depth_frame()
            depth_frame = depth_frame_og.reshape(424, 512)
            self.draw_infrared_frame(depth_frame, self.depth_surface)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:  self.status["offset"][0] += 50 * zoom_factor
                if event.key == pygame.K_RIGHT: self.status["offset"][0] -= 50 * zoom_factor
                if event.key == pygame.K_DOWN:  self.status["offset"][1] += 50 * zoom_factor
                if event.key == pygame.K_UP:    self.status["offset"][1] -= 50 * zoom_factor
                if event.key == pygame.K_EQUALS: self.status["scaling_factor"] += 0.5
                if event.key == pygame.K_MINUS:  self.status["scaling_factor"] -= 0.5
                if event.key == pygame.K_q:      self.status["rotate"][0] += math.pi / 8  # a
                if event.key == pygame.K_a:      self.status["rotate"][0] -= math.pi / 8  # q
                if event.key == pygame.K_w:      self.status["rotate"][1] += math.pi / 8  # z
                if event.key == pygame.K_s:      self.status["rotate"][1] -= math.pi / 8
                if event.key == pygame.K_e:      self.status["rotate"][2] += math.pi / 8
                if event.key == pygame.K_d:      self.status["rotate"][2] -= math.pi / 8
                print(event.key, self.status)

        if frame >= 1:
            xyz = []
            color_list = []

            step = 2
            width, height = 512, 424
            n_width, n_height = int(width / step), int(height / step)
            c_width, c_height = n_width * step, n_height * step

            map_time = time.time()

            y = np.repeat(np.arange(0, c_width, step), n_height)
            x = np.repeat(np.arange(0, c_width, step), n_height).reshape(n_height, n_width, order='F').ravel()
            z = depth_frame[0:c_height:step, 0:c_width:step].reshape(int(c_width * c_height / step ** 2), 1)
            np_depth_frame = np.c_[x, y, z]
            x = (np_depth_frame[:, 0] - width / 2) * np_depth_frame[:, 2] / 1000
            y = -(np_depth_frame[:, 1] - height / 2) * np_depth_frame[:, 2] / 1000
            xyz = np.c_[x, y, z / 4]

            self.debug_time["new_mapping"] += time.time() - map_time

            self.nodes = np.array(xyz)
            self.nodes = np.hstack((self.nodes, np.ones((len(self.nodes), 1))))

            transformation_matrices = []
            transform_time = time.time()

            # rotateXMatrix
            c = np.cos(self.status["rotate"][0])
            s = np.sin(self.status["rotate"][0])
            transformation_matrices.append(np.array([[1, 0, 0, 0],
                                                     [0, c, -s, 0],
                                                     [0, s, c, 0],
                                                     [0, 0, 0, 1]]))

            # rotateYMatrix
            c = np.cos(self.status["rotate"][1])
            s = np.sin(self.status["rotate"][1])
            transformation_matrices.append(np.array([[c, 0, s, 0],
                                                     [0, 1, 0, 0],
                                                     [-s, 0, c, 0],
                                                     [0, 0, 0, 1]]))

            # rotateZMatrix
            c = np.cos(self.status["rotate"][2])
            s = np.sin(self.status["rotate"][2])
            transformation_matrices.append(np.array([[c, -s, 0, 0],
                                                     [s, c, 0, 0],
                                                     [0, 0, 1, 0],
                                                     [0, 0, 0, 1]]))

            # scaling
            s = (self.status["scaling_factor"],) * 3
            transformation_matrices.append(np.array([[s[0], 0, 0, 0],
                                                     [0, s[1], 0, 0],
                                                     [0, 0, s[2], 0],
                                                     [0, 0, 0, 1]]))

            # translation
            transformation_matrices.append(np.array([[1, 0, 0, 0],
                                                     [0, 1, 0, 0],
                                                     [0, 0, 1, 0],
                                                     [self.status["offset"][0], self.status["offset"][1], 0, 1]]))

            for transform in transformation_matrices:
                self.nodes = np.dot(self.nodes, transform)

            self.debug_time["transforming"] += time.time() - transform_time

            display_time = time.time()

            self.node_surface.fill(black)
            color_to_draw = (white)

            for index, node in enumerate(self.nodes):
                if self.show_color_pixel: color_to_draw = color_list[index]
                pygame.draw.circle(self.node_surface, color_to_draw, (int(node[0]), -int(node[1])), 2, 0)

            self.debug_time["displaying"] += time.time() - display_time

        self.color_surface_to_draw = pygame.transform.scale(self.color_surface, (192 * factor, 108 * factor));
        self.depth_surface_to_draw = pygame.transform.scale(self.depth_surface, (192 * factor, 108 * factor));
        self.node_surface_to_draw = pygame.transform.scale(self.node_surface, (192 * factor * 2, 192 * factor * 2));
        self.screen.blit(self.color_surface_to_draw, (0, 0))
        self.screen.blit(self.depth_surface_to_draw, (192 * factor, 0))
        self.screen.blit(self.node_surface_to_draw, (0, 108 * factor))

        pygame.display.update()
        pygame.display.flip()

        if frame > 200:
            print(dict([(key, self.debug_time[key] / (time.time() - begin_time)) for key in self.debug_time.keys()]))
            break

