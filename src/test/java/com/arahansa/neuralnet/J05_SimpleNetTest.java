package com.arahansa.neuralnet;

import lombok.extern.slf4j.Slf4j;
import mikera.matrixx.Matrix;

import mikera.matrixx.impl.RowMatrix;
import mikera.vectorz.Vector;
import org.junit.Before;
import org.junit.Test;

import static org.assertj.core.api.Assertions.*;

/**
 * Created by jarvis on 2017. 4. 27..
 */
@Slf4j
public class J05_SimpleNetTest {


    J00_Helper helper;
    J01_CostFunction costFunction;
    J05_SimpleNet simpleNet;

    @Before
    public void setup(){
        helper = new J00_Helper();
        costFunction = new J01_CostFunction();
        simpleNet = new J05_SimpleNet(helper, costFunction);
    }

    @Test
    public void 초반값예측() throws Exception{
        Matrix w = Matrix.create(2,3);
        w.setElements(
                0.47355232, 0.9977393 , 0.84668094,
                0.85557411, 0.03563661,0.69422093
        );
        simpleNet.setW(w);

        Matrix x = Matrix.create(1,2);
        x.setElements(0.6, 0.9);
        final Matrix p = simpleNet.predict(x);
        log.info("p :{}", p);

        RowMatrix answer = new RowMatrix(Vector.of(1.054148091, 0.630716529, 1.132807401));
        assertThat(answer.equals(p)).isTrue();

        Matrix t = Matrix.create(1,3);
        t.setElements(0,0,1);
        final double loss = simpleNet.loss(x, t);
        log.info("loss : {}", loss);
        assertThat(loss).isEqualTo(0.9280, offset(0.0001));
    }

    @Test
    public void 심플넷_기울기_구하기() throws Exception{
        // X 설정
        Matrix x = Matrix.create(1,2);
        x.setElements(0.6, 0.9);

        // T 설정
        Matrix t = Matrix.create(1,3);
        t.setElements(0,0,1);

        // 초기 가중치 W 설정
        Matrix w = Matrix.create(2,3);
        w.setElements(
                0.47355232, 0.9977393 , 0.84668094,
                0.85557411, 0.03563661,0.69422093
        );
        simpleNet.setW(w);

        Matrix dW = simpleNet.numerical_gradient(x, t, w);
        log.info("dw :{}" , dW);

        Matrix answer = Matrix.create(2,3);
        answer.setElements(
                0.2192475712392561, 0.14356242984070455, -0.3628100010055757,
                0.3288713569016277,0.21534364482433954,-0.5442150014745017
        );
        assertThat(answer.equals(dW)).isTrue();
    }

}