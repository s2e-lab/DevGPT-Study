import org.apache.avro.Schema;
import org.apache.avro.generic.GenericData;
import org.apache.avro.generic.GenericRecord;
import org.apache.parquet.avro.AvroParquetWriter;
import org.apache.parquet.hadoop.metadata.CompressionCodecName;
import org.apache.parquet.io.OutputFile;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;

public class Main {

  public static void main(String[] args) throws IOException {

    // Define the Avro schema
    String rawSchema = "{\"type\":\"record\",\"name\":\"Test\",\"fields\":[{\"name\":\"field1\",\"type\":\"string\"}]}";
    Schema schema = new Schema.Parser().parse(rawSchema);

    // Create a record to write
    GenericRecord record = new GenericData.Record(schema);
    record.put("field1", "Test data");

    // Define the path and create the Parquet writer
    OutputFile outputFile = new OutputFile(Paths.get("data.parquet").toAbsolutePath(), Files::newOutputStream);
    try (var parquetWriter = AvroParquetWriter
        .<GenericRecord>builder(outputFile)
        .withSchema(schema)
        .withCompressionCodec(CompressionCodecName.SNAPPY)
        .build()) {

      // Write the record to the Parquet file
      parquetWriter.write(record);
    }
  }
}
