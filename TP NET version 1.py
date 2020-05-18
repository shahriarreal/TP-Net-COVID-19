def TP_net_1(input_shape):
    
    input_img = Input(shape = (32, 32, 3))
    tower_1 = Conv2D(64, (3,3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(weight_decay))(input_img)
    #     tower_1 = Conv2D(192, (5, 5), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), input_shape=x_train.shape[1:])
    tower_x = Conv2D(128, (3,3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(weight_decay))(tower_1)
    block1_output = GlobalAveragePooling2D()(tower_1)
    tower_y = MaxPooling2D(pool_size=(2, 2), padding='same')(tower_x)
    tower_y = BatchNormalization()(tower_y)
    tower_y = Dropout(0.5)(tower_y)
    # tower_z = Conv2D(192, (5,5), padding='same', activation='elu', kernel_regularizer=regularizers.l2(weight_decay))(tower_y)
    tower_a1 = Conv2D(192, (3,3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(weight_decay))(tower_y)
    tower_a = MaxPooling2D(pool_size=(2, 2), padding='same')(tower_a1)
    tower_a = BatchNormalization()(tower_a)
    tower_a = Dropout(0.5)(tower_a)
    
    tower_b1 = Conv2D(192, (3,3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(weight_decay))(tower_a)
    tower_b = MaxPooling2D(pool_size=(2, 2), padding='same')(tower_b1)
    tower_b = BatchNormalization()(tower_b)
    tower_b = Dropout(0.5)(tower_b)
    
    tower_c1 = Conv2D(256, (3,3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(weight_decay))(tower_b)
    tower_c = MaxPooling2D(pool_size=(2, 2), padding='same')(tower_c1)
    tower_c = BatchNormalization()(tower_c)
    tower_c = Dropout(0.5)(tower_c)
    
    
    
    
    
    tower_2 = AveragePooling2D(pool_size=(16, 16), padding='same')(tower_x)
    tower_2 = BatchNormalization()(tower_2)
    
    tower_3 = AveragePooling2D(pool_size=(8, 8), padding='same')(tower_a1)
    tower_3 = BatchNormalization()(tower_3)
    
    
    tower_4 = AveragePooling2D(pool_size=(4, 4), padding='same')(tower_b1)
    tower_4 = BatchNormalization()(tower_4)
    
    tower_5 = AveragePooling2D(pool_size=(2, 2), padding='same')(tower_c1)
    tower_5 = BatchNormalization()(tower_5)
    
    
    
    
    
    
    
    
    
    
    output = keras.layers.concatenate([tower_c, tower_2, tower_3,tower_4,tower_5], axis = 3)
    #     output = Flatten()(output)
    output = GlobalAveragePooling2D()(output)
    out1 = keras.layers.concatenate([output,block1_output],axis=1)
    
    out    = Dense(10, activation='softmax')(out1)
    
    
    
    
    
    
    
    model = Model(inputs = input_img, outputs = out)
    print(model.summary())
    return model
