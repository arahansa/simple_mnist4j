package com.arahansa.neuralnet;

import lombok.extern.slf4j.Slf4j;
import mikera.matrixx.Matrix;
import org.junit.Before;
import org.junit.Test;

import static org.assertj.core.api.Assertions.*;

/**
 * Created by jarvis on 2017. 4. 27..
 */
@Slf4j
public class J01_CostFunctionTest {

    J01_CostFunction costFunction;

    @Before
    public void setup(){
        costFunction = new J01_CostFunction();
    }

    @Test
    public void 로그테스트(){
        Matrix answer = new Matrix(2,2);
        answer.setElements(1,2,3,4);
        answer.log();
        log.info("answer : {}", answer);
    }


    // 파이썬 값과 비교한 테스트
    @Test
    public void 교차엔트로피테스트(){
        Matrix m1 = new Matrix(4, 1);
        m1.setElements(0.0, 0.9, 0.1, 0.0);
        Matrix m2 = new Matrix(4, 1);
        m2.setElements(0, 1, 0, 0);

        double crossEntropyErr = costFunction.getCrossEntropyErr(m1, m2);
        assertThat(crossEntropyErr).isEqualTo(0.10536, offset(0.5));
    }


    @Test
    public void costErr() throws Exception{

        Matrix m = Matrix.create(3,3);

        m.setElements(0.00235587 , 0.04731414,  0.95032999,
                0.00235587 , 0.04731414,  0.95032999,
                0.00235587 , 0.04731414,  0.95032999);


        Matrix t = Matrix.create(3,3);
        t.setElements(1.0,0.0,0.0, 1.0,0.0,0.0, 1.0,0.0,0.0);

        double crossEntropyErr = J01_CostFunction.getCrossEntropyErr4Batch(m, t);
        log.info("cross ent  : {}", crossEntropyErr);


        assertThat(crossEntropyErr).isEqualTo(6.05, offset(0.001));
    }
}