package com.arahansa.neuralnet;

import com.arahansa.data.Grad;
import com.arahansa.image.LoadMnist;
import lombok.extern.slf4j.Slf4j;
import mikera.matrixx.Matrix;
import org.junit.Test;
import org.springframework.util.StopWatch;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.assertj.core.api.Assertions.*;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.offset;


/**
 * Created by arahansa on 2017-05-03.
 */
@Slf4j
public class J06_TwoLayerNetTest {

    @Test
    public void 미니배치_학습구현() throws Exception{
        StopWatch watchB = new StopWatch();
        watchB.start();
        LoadMnist loadMnist = new LoadMnist();
        Map<String, Matrix> mnistMap = loadMnist.loadMnist();


        Matrix x_train = mnistMap.get("x_train");
        Matrix t_train = mnistMap.get("t_train");
        Matrix x_test = mnistMap.get("x_test");
        Matrix t_test = mnistMap.get("t_test");

        int iters_num = 1000;
        int batch_size = 100;
        int train_size = x_train.getShape(0);
        double learning_rate  = 1.0;
        int iter_per_epoch = train_size / batch_size;

        List<Double> train_loss_list = new ArrayList<>(iters_num);
        List<Double> train_acc_list = new ArrayList<>(iter_per_epoch);
        List<Double> test_acc_list = new ArrayList<>(iter_per_epoch);



        J06_TwoLayerNet network = new J06_TwoLayerNet(15, 50, 10, 1.0f);


        for(int i=0;i<iters_num;i++){
            log.info("count : {}", i);
            StopWatch watch = new StopWatch();
            watch.start();
            Map<String, Matrix> bm = J00_Helper.getBatchMatrix(x_train, t_train, batch_size);
            Matrix x_batch = bm.get("x_batch");
            Matrix t_batch = bm.get("t_batch");
            Grad grad = network.numerical_gradient(x_batch, t_batch);
            network.renewParams(grad, learning_rate);


            Double loss = network.lossFunc.apply(x_batch, t_batch);
            train_loss_list.add(loss);

            if( i % iter_per_epoch == 0){
                double train_acc = network.accuracy(x_train, t_train);
                double test_acc = network.accuracy(x_test, t_test);

                train_acc_list.add(train_acc);
                test_acc_list.add(test_acc);
                log.info("i : {} , train acc : {}, test acc : {}", i, train_acc, test_acc);
            }
            watch.prettyPrint();
        }

        System.out.println(watchB.prettyPrint());
    }


    @Test
    public void accuracyTest() throws Exception{
        J06_TwoLayerNet network = new J06_TwoLayerNet(10, 10, 10, 0.01f);

        mikera.vectorz.Vector v = mikera.vectorz.Vector.of(1, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        Matrix x = Matrix.create(10, 10);
        Matrix t = Matrix.create(10, 10);

        x.add(v);
        t.add(v);

        double accuracy = network.calcul_accracy(x, t);
        assertThat(accuracy).isEqualTo(1.0, offset(0.1));

        t.set(9, 0, 0.0);
        t.set(9, 1, 1.0);
        accuracy = network.calcul_accracy(x, t);
        assertThat(accuracy).isEqualTo(0.9, offset(0.01));
    }

    @Test
    public void addRowTest() throws Exception{
        mikera.vectorz.Vector v = mikera.vectorz.Vector.of(1, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        Matrix x = Matrix.create(10, 10);

        x.add(v);

        log.info("v: {}", x);
        x.add(v);

        log.info("v: {}", x);
    }



}