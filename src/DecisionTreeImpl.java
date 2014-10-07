import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Fill in the implementation details of the class DecisionTree using this file.
 * Any methods or secondary classes that you want are fine but we will only
 * interact with those methods in the DecisionTree framework.
 * 
 * You must add code for the 5 methods specified below.
 * 
 * See DecisionTree for a description of default methods.
 */
public class DecisionTreeImpl extends DecisionTree {
	private DecTreeNode root;
	private List<String> labels; // ordered list of class labels
	private List<String> attributes; // ordered list of attributes
	private Map<String, List<String>> attributeValues; // map to ordered
														// discrete values taken
														// by attributes

	/**
	 * Answers static questions about decision trees.
	 */
	DecisionTreeImpl() {
		// no code necessary
		// this is void purposefully
	}

	/**
	 * Build a decision tree given only a training set.
	 * 
	 * @param train: the training set
	 */
	DecisionTreeImpl(DataSet train) {

		this.labels = train.labels;
		this.attributes = train.attributes;
		this.attributeValues = train.attributeValues;
		// TODO: add code here
	}

	/**
	 * Build a decision tree given a training set then prune it using a tuning
	 * set.
	 * 
	 * @param train: the training set
	 * @param tune: the tuning set
	 */
	DecisionTreeImpl(DataSet train, DataSet tune) {

		this.labels = train.labels;
		this.attributes = train.attributes;
		this.attributeValues = train.attributeValues;
		// TODO: add code here
	}

	@Override
	public String classify(Instance instance) {

		// TODO: add code here
		return null;
	}

	@Override
	/**
	 * Print the decision tree in the specified format
	 */
	public void print() {

		printTreeNode(root, null, 0);
	}
	
	/**
	 * Prints the subtree of the node
	 * with each line prefixed by 4 * k spaces.
	 */
	public void printTreeNode(DecTreeNode p, DecTreeNode parent, int k) {
		StringBuilder sb = new StringBuilder();
		for (int i = 0; i < k; i++) {
			sb.append("    ");
		}
		String value;
		if (parent == null) {
			value = "ROOT";
		} else{
			String parentAttribute = attributes.get(parent.attribute);
			value = attributeValues.get(parentAttribute).get(p.parentAttributeValue);
		}
		sb.append(value);
		if (p.terminal) {
			sb.append(" (" + labels.get(p.label) + ")");
			System.out.println(sb.toString());
		} else {
			sb.append(" {" + attributes.get(p.attribute) + "?}");
			System.out.println(sb.toString());
			for(DecTreeNode child: p.children) {
				printTreeNode(child, p, k+1);
			}
		}
	}

	@Override
	public void rootInfoGain(DataSet train) {

		this.labels = train.labels;
		this.attributes = train.attributes;
		this.attributeValues = train.attributeValues;

		int instance_count = train.instances.size();
		int label_counts[] = new int[labels.size()];

		for (Instance instance : train.instances) {
			label_counts[instance.label]++;
		}

		double totalEntropy = calculateEntropy(label_counts, instance_count);

		for (int i = 0; i < attributes.size(); i++) {
			// for the ith attribute
			int number_of_attribute_values = attributeValues.get(
					attributes.get(i)).size();
			int attribute_counts[] = new int[number_of_attribute_values];
			Map<Integer, int[]> map = new HashMap<>();
			for (int j = 0; j < number_of_attribute_values; j++) {
				map.put(j, new int[labels.size()]);
			}

			for (Instance instance : train.instances) {
				int attributeValue = instance.attributes.get(i);
				attribute_counts[attributeValue]++;
				map.get(attributeValue)[instance.label]++;
			}

			double conditionalEntropy = 0;
			for (int j = 0; j < number_of_attribute_values; j++) {
				int attribute_count = attribute_counts[j];
				double p = (double) attribute_count / instance_count;
				conditionalEntropy = conditionalEntropy + p
						* calculateEntropy(map.get(j), attribute_count);
			}
			System.out
					.println(attributes.get(i)
							+ " "
							+ String.format("%.5f",
									(totalEntropy - conditionalEntropy)));
		}
	}

	private double calculateEntropy(int[] label_counts, int total_count) {
		double result = 0;
		for (int label_count : label_counts) {
			double p = (double) label_count / total_count;
			if (p != 0)
				result = result - p * Math.log(p) / Math.log(2);
		}
		return result;
	}
	
	
	
}
