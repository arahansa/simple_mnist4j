package com.arahansa.image;

import lombok.extern.slf4j.Slf4j;
import mikera.matrixx.Matrix;
import org.junit.Before;
import org.junit.Test;

import java.util.Map;

import static org.assertj.core.api.Assertions.*;

/**
 * Created by arahansa on 2017-05-03.
 */
@Slf4j
public class LoadMnistTest {

    LoadMnist loadMnist;

    @Before
    public void setup(){
        loadMnist= new LoadMnist();
    }

    @Test
    public void getTtrainTest() throws Exception{
        final Map<Integer, Matrix> ttrainSampleMap = loadMnist.getTtrainSampleMap();
        log.info("ttrainSampleMap : {}", ttrainSampleMap);
        Matrix m0 = Matrix.create(1, 10);
        m0.setElements(1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0);

        Matrix m9 = Matrix.create(1, 10);
        m9.setElements(0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0);

        assertThat(ttrainSampleMap.get(0).equals(m0)).isTrue();
        assertThat(ttrainSampleMap.get(9).equals(m9)).isTrue();
    }

    @Test
    public void getXTrainTest() throws Exception{
        Map<Integer, Matrix> xtrainSampleMap = loadMnist.getXtrainSampleMap();
        log.info("xtrainSampleMap : {}", xtrainSampleMap);

        Matrix m0 = Matrix.create(1, 15);
        m0.setElements(1.0,1.0,1.0,1.0,0.0,1.0,1.0,0.0,1.0,1.0,0.0,1.0,1.0,1.0,1.0);
        Matrix m9 = Matrix.create(1, 15);
        m9.setElements(1.0,1.0,1.0,1.0,0.0,1.0,1.0,1.0,1.0,0.0,0.0,1.0,0.0,0.0,1.0);

        assertThat(xtrainSampleMap.get(0).equals(m0)).isTrue();
        assertThat(xtrainSampleMap.get(9).equals(m9)).isTrue();
    }

    @Test
    public void 나누기테스트() throws Exception{

        for(int i=0;i<20;i++){
            log.info("i / 10 : {}", i%10);
        }
    }


}