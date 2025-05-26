package com.ai.churnprediction.trainmodel.api.nd4j;
import com.ai.churnprediction.trainmodel.api.datavec.DatavecUtility;
import org.datavec.api.transform.TransformProcess;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.File;

public class ND4JUtility {
    // Step 5: Apply final transform
    public static DataSet getDataSet(TransformProcess fullTransform , File cleanedCsv) throws Exception {
        DataSetIterator fullIterator = DatavecUtility.applyFinalTransform(fullTransform,cleanedCsv);
        DataSet allData = fullIterator.next();
        allData.shuffle();
        return allData;
    }
}
