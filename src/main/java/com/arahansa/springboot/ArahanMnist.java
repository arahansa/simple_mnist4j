package com.arahansa.springboot;

/**
 * Created by arahansa on 2017-05-06.
 */

import com.arahansa.data.Grad;
import com.arahansa.image.LoadMnist;
import com.arahansa.neuralnet.J06_TwoLayerNet;
import jdk.nashorn.internal.objects.annotations.Constructor;
import lombok.extern.slf4j.Slf4j;
import mikera.matrixx.Matrix;
import mikera.vectorz.impl.ArraySubVector;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.ResponseBody;

import javax.annotation.PostConstruct;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.stream.IntStream;

@SpringBootApplication
public class ArahanMnist {

    public static void main(String[] args) {
        SpringApplication.run(ArahanMnist.class, args);
    }

    @Slf4j
    @Controller
    public static class HelloController{

        private J06_TwoLayerNet network;
        private Map<Integer, Matrix> xtrainSampleMap;
        private Map<Integer, Matrix> ttrainSampleMap;

        @PostConstruct
        public void setup(){
            double learning_rate = 0.2;
            network = new J06_TwoLayerNet(15, 20, 10, new Float(learning_rate));
            LoadMnist loadMnist = new LoadMnist();
            xtrainSampleMap = loadMnist.getXtrainSampleMap();
            ttrainSampleMap = loadMnist.getTtrainSampleMap();


            int size = 10;
            Matrix x_batch = Matrix.create(size, 15);
            Matrix t_batch = Matrix.create(size, 10);

            for(int i=0;i<size;i++){
                ArraySubVector xRow = xtrainSampleMap.get(i % 10).getRow(0);
                ArraySubVector tRow = ttrainSampleMap.get(i % 10).getRow(0);
                x_batch.setRow(i, xRow);
                t_batch.setRow(i, tRow);
            }

            Grad grad = network.numerical_gradient(x_batch, t_batch);
            network.renewParams(grad, 1.0);

            for(int i=0;i<1000;i++){
                grad = network.numerical_gradient(x_batch, t_batch);
                network.renewParams(grad, learning_rate);

                final double accuracy = network.accuracy(x_batch, t_batch);
                if(accuracy>=0.99) break;
            }
            log.info("학습완료 디기딩동!");
        }

        @GetMapping("/")
        public String hello(){
            return "index";
        }

        @PostMapping("/draw")
        public String draw(int[] check, Model model){
           log.info("pixel : {}", check);
            Matrix m = getMatrixbyCheck(check);
            log.info("m :{}", m);
            Matrix p = network.predict(m);
            log.info("predict : {}", p);
            log.info("predict number :{}", p.getRow(0).maxElementIndex());
            model.addAttribute("p", p.getRow(0).maxElementIndex());
            return "index";
        }

        @PostMapping("/real-education")
        public String realEdu(int[] check, int realanswer){
            log.info("pixel : {}", check);
            log.info("realanswer :{}", realanswer);
            Matrix m = getMatrixbyCheck(check);
            Matrix t_batch = Matrix.create(1, 10);
            final ArraySubVector row = ttrainSampleMap.get(realanswer).getRow(0);
            t_batch.setRow(0, row);

            for(int i=0;i<1000;i++){
                Grad grad = network.numerical_gradient(m, t_batch);
                network.renewParams(grad, 0.1);
            }
            return "redirect:/";
        }

        private Matrix getMatrixbyCheck(int[] check){
            Matrix m = Matrix.create(1, 15);
            m.setElements(IntStream.of(check).mapToDouble(Double::new).toArray());
            return m;
        }

    }

}
