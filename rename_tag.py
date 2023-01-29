import tensorflow as tf

# Create a summary writer
writer = tf.summary.create_file_writer("path/to/logs")

# Write a scalar to the summary
with writer.as_default():
    tf.summary.scalar("original_name", 0.5, step=0)

# Close the writer
writer.close()

# Open the summary file
summary_proto = tf.summary.summary_iterator("path/to/logs").get_next()

# Find the scalar and rename its title
for value in summary_proto.value:
    if value.tag == "original_name":
        value.tag = "new_name"

# Re-write the summary file
with open("path/to/logs", "wb") as f:
    f.write(summary_proto.SerializeToString())
