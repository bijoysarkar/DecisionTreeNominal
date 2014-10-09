import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Queue;
import java.util.Set;

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
	 * @param train
	 *            : the training set
	 */
	DecisionTreeImpl(DataSet train) {

		this.labels = train.labels;
		this.attributes = train.attributes;
		this.attributeValues = train.attributeValues;
		int maxLabel = getMaxLabel(getLabelCounts(train.instances));
		this.root = buildTree(train.instances, new HashSet<String>(), -1,
				maxLabel);

	}

	/**
	 * Build a decision tree given a training set then prune it using a tuning
	 * set.
	 * 
	 * @param train
	 *            : the training set
	 * @param tune
	 *            : the tuning set
	 */
	DecisionTreeImpl(DataSet train, DataSet tune) {

		this.labels = train.labels;
		this.attributes = train.attributes;
		this.attributeValues = train.attributeValues;
		int maxLabel = getMaxLabel(getLabelCounts(train.instances));
		this.root = buildTree(train.instances, new HashSet<String>(), -1,
				maxLabel);
		// TODO: add code here
		prune(root, tune.instances);
	}

	@Override
	public String classify(Instance instance) {
		DecTreeNode current_node = root;
		while (!current_node.terminal) {
			current_node = current_node.children.get(instance.attributes
					.get(current_node.attribute));
		}
		return labels.get(current_node.label);
	}

	@Override
	/**
	 * Print the decision tree in the specified format
	 */
	public void print() {
		printTreeNode(root, null, 0);
	}

	/**
	 * Prints the subtree of the node with each line prefixed by 4 * k spaces.
	 */
	public void printTreeNode(DecTreeNode p, DecTreeNode parent, int k) {
		StringBuilder sb = new StringBuilder();
		for (int i = 0; i < k; i++) {
			sb.append("    ");
		}
		String value;
		if (parent == null) {
			value = "ROOT";
		} else {
			String parentAttribute = attributes.get(parent.attribute);
			value = attributeValues.get(parentAttribute).get(
					p.parentAttributeValue);
		}
		sb.append(value);
		if (p.terminal) {
			sb.append(" (" + labels.get(p.label) + ")");
			System.out.println(sb.toString());
		} else {
			sb.append(" {" + attributes.get(p.attribute) + "?}");
			System.out.println(sb.toString());
			for (DecTreeNode child : p.children) {
				printTreeNode(child, p, k + 1);
			}
		}
	}

	private DecTreeNode buildTree(List<Instance> instanceList,
			Set<String> processedAttributes, int parentAttributeValue,
			int parentMaxLabel) {

		// If empty examples return default
		if (instanceList.size() == 0) {
			return new DecTreeNode(parentMaxLabel, null, parentAttributeValue,
					true);
		}

		int label_counts[] = getLabelCounts(instanceList);
		int maxLabel = getMaxLabel(label_counts);
		// If examples have same label return y
		if (isAllSameLabels(label_counts)) {
			return new DecTreeNode(maxLabel, null, parentAttributeValue, true);
		}

		// If empty question then return majority vote in examples
		if (processedAttributes.size() == attributes.size()) {
			return new DecTreeNode(maxLabel, null, parentAttributeValue, true);
		}

		// Find best question
		Map<Integer, Double> infoGain = calculateRootInfoGain(
				processedAttributes, instanceList);
		double maxInfoGain = Double.MIN_VALUE;
		Integer maxInfoGainAttribute = null;
		for (Entry<Integer, Double> entry : infoGain.entrySet()) {
			if (entry.getValue() > maxInfoGain) {
				maxInfoGain = entry.getValue();
				maxInfoGainAttribute = entry.getKey();
			}
		}

		DecTreeNode decTreeNode = new DecTreeNode(maxLabel,
				maxInfoGainAttribute, parentAttributeValue, false);

		Set<String> processedAttributesCopy = new HashSet<String>(
				processedAttributes);
		processedAttributesCopy.add(attributes.get(maxInfoGainAttribute));

		List<String> attributeValuesList = attributeValues.get(attributes
				.get(maxInfoGainAttribute));
		for (int i = 0; i < attributeValuesList.size(); i++) {
			List<Instance> l = partitionInstances(instanceList,
					maxInfoGainAttribute, i);
			decTreeNode.addChild(buildTree(l, processedAttributesCopy, i,
					maxLabel));
		}

		return decTreeNode;
	}

	private void prune(DecTreeNode rootNode, List<Instance> tuneSet) {
		while (pruneIteration(rootNode, tuneSet))
			;
	}

	private boolean pruneIteration(DecTreeNode rootNode, List<Instance> tuneSet) {

		double initial_accuracy = calculateAccuracy(tuneSet);
		double max_accuracy = Double.MIN_NORMAL;
		DecTreeNode prune_node = null;

		// Iterate, set to terminal, get accuracy and unset terminal at each
		// internal node
		// Keep a pointer with maximum accuracy till now
		// End of each full traversal actually prune the node with maximum
		// accuracy on pruning
		List<DecTreeNode> queue = new ArrayList<DecTreeNode>();
		queue.add(rootNode);
		// Since tree so no visited marking required
		while (!queue.isEmpty()) {
			DecTreeNode decTreeNode = queue.remove(0);
			if (!decTreeNode.terminal) {
				decTreeNode.terminal = true;
				double prune_accuracy = calculateAccuracy(tuneSet);
				if (prune_accuracy > max_accuracy) {
					max_accuracy = prune_accuracy;
					prune_node = decTreeNode;
				}
				decTreeNode.terminal = false;
				queue.addAll(decTreeNode.children);
			}
		}

		if (max_accuracy >= initial_accuracy && prune_node != null) {
			prune_node.terminal = true;
			prune_node.children.clear();
			return true;
		}
		return false;
	}

	@Override
	public void rootInfoGain(DataSet train) {

		this.labels = train.labels;
		this.attributes = train.attributes;
		this.attributeValues = train.attributeValues;
		Map<Integer, Double> infoGain = calculateRootInfoGain(
				new HashSet<String>(), train.instances);
		for (int i = 0; i < attributes.size(); i++) {
			System.out.println(attributes.get(i) + " "
					+ String.format("%.5f", infoGain.get(i)));
		}
	}

	private Map<Integer, Double> calculateRootInfoGain(
			Set<String> processedAttributes, List<Instance> instancesList) {

		Map<Integer, Double> result = new HashMap<Integer, Double>();

		// Total number of instances in this set
		int instance_count = instancesList.size();
		// Get label counts of the instances and calculate the entropy
		double totalEntropy = calculateEntropy(getLabelCounts(instancesList),
				instance_count);

		// Iterate over all the attributes
		for (int i = 0; i < attributes.size(); i++) {
			// For the ith attribute
			String attributeName = attributes.get(i);
			// Skip if this attribute has already been answered
			// This means that all the instances in this partition set
			// have a certain value of this attribute so no entropy
			// ie 0 information gain by asking the same question
			if (processedAttributes.contains(attributeName))
				continue;

			// For this attribute get the possible number of values
			int number_of_attribute_values = attributeValues.get(attributeName)
					.size();
			// Set counter for the frequency of values of this attribute
			int attribute_counts[] = new int[number_of_attribute_values];
			// Create map to store the frequency of labels corresponding to each
			// values
			Map<Integer, int[]> map = new HashMap<>();
			for (int j = 0; j < number_of_attribute_values; j++) {
				map.put(j, new int[labels.size()]);
			}

			for (Instance instance : instancesList) {
				// Get the value for the current attribute (ith attribute for
				// this iteration)
				int attributeValue = instance.attributes.get(i);
				// Increment frequency of this value in the counter
				attribute_counts[attributeValue]++;
				// Increment the frequency in the map for the label for this
				// attribute value
				map.get(attributeValue)[instance.label]++;
			}

			// System.out.println("attribute_counts "+Arrays.toString(attribute_counts));
			double conditionalEntropy = 0;
			// Go over all the possible values of the attribute
			for (int j = 0; j < number_of_attribute_values; j++) {
				int attribute_count = attribute_counts[j];
				if (attribute_count > 0) {
					double p = (double) attribute_count / instance_count;
					conditionalEntropy = conditionalEntropy + p
							* calculateEntropy(map.get(j), attribute_count);
				}
			}
			result.put(i, totalEntropy - conditionalEntropy);
		}
		return result;
	}

	private int[] getLabelCounts(List<Instance> instancesList) {
		int label_counts[] = new int[labels.size()];

		for (Instance instance : instancesList) {
			label_counts[instance.label]++;
		}
		return label_counts;
	}

	private boolean isAllSameLabels(int label_counts[]) {
		int non_zero = 0;
		for (int count : label_counts)
			if (count != 0)
				non_zero++;
		return (non_zero == 1);
	}

	private int getMaxLabel(int label_counts[]) {
		int max_index = 0;
		int max = label_counts[max_index];
		for (int i = 1; i < label_counts.length; i++) {
			if (label_counts[i] > max) {
				max_index = i;
				max = label_counts[i];
			}
		}
		return max_index;
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

	private double calculateAccuracy(List<Instance> instanceList) {
		int correct_count = 0;
		for (Instance instance : instanceList) {
			if (labels.get(instance.label).equals(classify(instance)))
				correct_count++;
		}
		return (double) correct_count / instanceList.size();
	}

	private List<Instance> partitionInstances(List<Instance> instanceList,
			int attributeNumber, int attributeValueIndex) {
		List<Instance> list = new ArrayList<Instance>();
		for (Instance instance : instanceList) {
			if (instance.attributes.get(attributeNumber) == attributeValueIndex)
				list.add(instance);
		}
		return list;
	}

}
