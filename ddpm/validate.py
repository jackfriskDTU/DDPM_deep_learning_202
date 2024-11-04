def validate(model, validation_data):
    model.eval()
    with torch.no_grad():
        for data in validation_data:
            inputs, labels = data
            outputs = model(inputs)
            # For simplicity, just print the outputs
            print(f"Validation outputs: {outputs}")