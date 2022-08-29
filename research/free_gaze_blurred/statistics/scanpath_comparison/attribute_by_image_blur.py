

# calculates the number of a given attribute (i.e., saccades, blinks, atc.) appearances per image blur level
def attribute_per_image_blur(attribute_df, images_df):

    # add image_blur column to attribute_df
    for index, row in attribute_df.iterrows():
        attribute_df.loc[index, 'image_blur'] = images_df.loc[row['image_index'], 'blur']
    attribute_df = attribute_df.astype({"image_index": int}, errors='raise')

    # calc number of attribute appearances per blur level
    attributes_per_blur = {}
    for index, row in attribute_df.iterrows():
        if row['image_blur'] in attributes_per_blur:
            attributes_per_blur[row['image_blur']] += 1
        else:
            attributes_per_blur[row['image_blur']] = 0

    # normalize 0.0 blur level
    zero_blur_count = 0
    for index, row in images_df.iterrows():
        if row['blur'] == 0.0:
            zero_blur_count += 1
    attributes_per_blur[0.0] = attributes_per_blur[0.0] / zero_blur_count

    return attributes_per_blur
